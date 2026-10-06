use rayon::prelude::*;

use crate::cli::Args;
use crate::{ESCAPE_R2, Frame, Renderer, colour};

pub struct CpuRenderer {
    width: usize,
    height: usize,
    cre: f64,
    cim: f64,
    orbit: Option<Vec<f64>>,
    pool: rayon::ThreadPool,
}

impl CpuRenderer {
    pub fn new(args: &Args, cre: f64, cim: f64, orbit: Option<Vec<f64>>) -> anyhow::Result<Self> {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(args.threads).build()?;
        Ok(Self { width: args.width as usize, height: args.height as usize, cre, cim, orbit, pool })
    }
}

impl Renderer for CpuRenderer {
    fn render(&mut self, f: Frame, rgb: &mut [u8]) -> anyhow::Result<()> {
        let (w, h) = (self.width, self.height);
        let (cre, cim, orbit) = (self.cre, self.cim, self.orbit.as_deref());
        // One row per task; rayon's work stealing balances the uneven rows.
        self.pool.install(|| {
            rgb.par_chunks_mut(w * 3).enumerate().for_each(|(y, row)| {
                let dci = (0.5 * h as f64 - (y as f64 + 0.5)) * f.spacing;
                match orbit {
                    Some(r) => row_perturb(row, w, r, dci, f.spacing, f.max_iter),
                    None => row_f64(row, w, cre, cim + dci, f.spacing, f.max_iter),
                }
            });
        });
        Ok(())
    }
}

// Each task iterates LANES pixels of a row at once. One pixel's iterations form a serial
// dependency chain (each step needs the previous z), so a single pixel leaves the core
// mostly idle; independent lanes fill it, and LLVM packs them into SIMD registers. When a
// lane finishes, it is coloured and refilled with the row's next pixel. Every lane does
// exactly the scalar operations of the one-pixel loop, so the output is unchanged.
// Lane state is all f64 (the iteration count too, exact up to 2^53) and the per-iteration
// loops have no branches, so they vectorize.
const LANES: usize = 8;

/// Writes the colour of a finished lane: escaped, or black at max_iter.
#[inline]
fn finish(row: &mut [u8], px: usize, n: f64, r2: f64) {
    let p = &mut row[3 * px..3 * px + 3];
    if r2 > ESCAPE_R2 { colour(p, n as i32, r2) } else { p.fill(0) }
}

/// Hands out a row's pixels to lanes.
struct Feed {
    next: usize,
    width: usize,
    active: usize,
}

impl Feed {
    /// The next pixel's index, or None when the row is used up.
    fn take(&mut self) -> Option<usize> {
        (self.next < self.width).then(|| {
            self.next += 1;
            self.active += 1;
            self.next - 1
        })
    }
}

const IDLE: usize = usize::MAX;

/// f64 lane state: z <- z^2 + c from z = 0.
struct F64Lanes {
    zr: [f64; LANES],
    zi: [f64; LANES],
    cr: [f64; LANES],
    ci: [f64; LANES],
    r2: [f64; LANES],
    n: [f64; LANES],
    lim: [f64; LANES],
    px: [usize; LANES],
}

fn row_f64(row: &mut [u8], w: usize, cre: f64, ci_row: f64, spacing: f64, max_iter: i32) {
    let mut l = F64Lanes {
        zr: [0.0; LANES],
        zi: [0.0; LANES],
        cr: [0.0; LANES],
        ci: [0.0; LANES],
        r2: [0.0; LANES],
        n: [0.0; LANES],
        lim: [0.0; LANES],
        px: [IDLE; LANES],
    };
    let mut feed = Feed { next: 0, width: w, active: 0 };
    let load = |l: &mut F64Lanes, feed: &mut Feed, k: usize| {
        (l.zr[k], l.zi[k], l.n[k]) = (0.0, 0.0, 0.0);
        match feed.take() {
            Some(x) => {
                let dcr = (x as f64 + 0.5 - 0.5 * w as f64) * spacing;
                (l.cr[k], l.ci[k], l.lim[k], l.px[k]) = (cre + dcr, ci_row, max_iter as f64, x);
            }
            // idle lane: c = 0 never escapes
            None => (l.cr[k], l.ci[k], l.lim[k], l.px[k]) = (0.0, 0.0, f64::INFINITY, IDLE),
        }
    };
    for k in 0..LANES {
        load(&mut l, &mut feed, k);
    }
    while feed.active > 0 {
        let mut any = false;
        for k in 0..LANES {
            let (zr2, zi2) = (l.zr[k] * l.zr[k], l.zi[k] * l.zi[k]);
            let ni = 2.0 * l.zr[k] * l.zi[k] + l.ci[k];
            let nr = zr2 - zi2 + l.cr[k];
            let rk = nr * nr + ni * ni;
            (l.zr[k], l.zi[k], l.r2[k]) = (nr, ni, rk);
            l.n[k] += 1.0;
            any |= (rk > ESCAPE_R2) | (l.n[k] >= l.lim[k]);
        }
        if !any {
            continue;
        }
        for k in 0..LANES {
            if l.px[k] == IDLE || !(l.r2[k] > ESCAPE_R2 || l.n[k] >= l.lim[k]) {
                continue;
            }
            finish(row, l.px[k], l.n[k], l.r2[k]);
            feed.active -= 1;
            load(&mut l, &mut feed, k);
        }
    }
}

/// deep lane state: dz <- (2Z + dz) dz + dc around the reference orbit, with rebasing.
struct PerturbLanes {
    dzr: [f64; LANES],
    dzi: [f64; LANES],
    dcr: [f64; LANES],
    r2: [f64; LANES],
    n: [f64; LANES],
    lim: [f64; LANES],
    m: [usize; LANES],
    px: [usize; LANES],
}

fn row_perturb(row: &mut [u8], w: usize, r: &[f64], dci: f64, spacing: f64, max_iter: i32) {
    let (r, _) = r.as_chunks::<2>(); // (re, im) pairs
    let last = r.len() - 1;
    let mut l = PerturbLanes {
        dzr: [0.0; LANES],
        dzi: [0.0; LANES],
        dcr: [0.0; LANES],
        r2: [0.0; LANES],
        n: [0.0; LANES],
        lim: [0.0; LANES],
        m: [0; LANES],
        px: [IDLE; LANES],
    };
    let mut feed = Feed { next: 0, width: w, active: 0 };
    let load = |l: &mut PerturbLanes, feed: &mut Feed, k: usize| {
        (l.dzr[k], l.dzi[k], l.n[k], l.m[k]) = (0.0, 0.0, 0.0, 0);
        match feed.take() {
            Some(x) => {
                let dcr = (x as f64 + 0.5 - 0.5 * w as f64) * spacing;
                (l.dcr[k], l.lim[k], l.px[k]) = (dcr, max_iter as f64, x);
            }
            // idle lane: follows the reference orbit and is never checked
            None => (l.dcr[k], l.lim[k], l.px[k]) = (0.0, f64::INFINITY, IDLE),
        }
    };
    for k in 0..LANES {
        load(&mut l, &mut feed, k);
    }
    while feed.active > 0 {
        // Three passes, so the arithmetic is a straight-line block LLVM can vectorize:
        // load each lane's Z_m and Z_m+1 (different indices per lane), step, update m.
        let (mut zm, mut zn, mut end) = ([[0.0f64; 2]; LANES], [[0.0f64; 2]; LANES], [false; LANES]);
        for k in 0..LANES {
            let m = l.m[k];
            (zm[k], zn[k], end[k]) = (r[m], r[m + 1], m + 1 == last);
        }
        let mut any = false;
        let mut rebase = [false; LANES];
        for k in 0..LANES {
            let tr = 2.0 * zm[k][0] + l.dzr[k];
            let ti = 2.0 * zm[k][1] + l.dzi[k];
            let nr = tr * l.dzr[k] - ti * l.dzi[k] + l.dcr[k];
            let ni = tr * l.dzi[k] + ti * l.dzr[k] + dci;
            let (zr, zi) = (zn[k][0] + nr, zn[k][1] + ni);
            let rk = zr * zr + zi * zi;
            l.r2[k] = rk;
            l.n[k] += 1.0;
            any |= (rk > ESCAPE_R2) & (l.px[k] != IDLE) | (l.n[k] >= l.lim[k]);
            // Rebase (Zhuoran): restart the reference when z gets closer to 0 than dz.
            rebase[k] = (rk < nr * nr + ni * ni) | end[k];
            l.dzr[k] = if rebase[k] { zr } else { nr };
            l.dzi[k] = if rebase[k] { zi } else { ni };
        }
        for k in 0..LANES {
            l.m[k] = if rebase[k] { 0 } else { l.m[k] + 1 };
        }
        if !any {
            continue;
        }
        for k in 0..LANES {
            if l.px[k] == IDLE || !(l.r2[k] > ESCAPE_R2 || l.n[k] >= l.lim[k]) {
                continue;
            }
            finish(row, l.px[k], l.n[k], l.r2[k]);
            feed.active -= 1;
            load(&mut l, &mut feed, k);
        }
    }
}
