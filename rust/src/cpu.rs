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
                for (x, px) in row.chunks_exact_mut(3).enumerate() {
                    let dcr = (x as f64 + 0.5 - 0.5 * w as f64) * f.spacing;
                    match orbit {
                        Some(r) => pixel_perturb(px, r, dcr, dci, f.max_iter),
                        None => pixel_f64(px, cre + dcr, cim + dci, f.max_iter),
                    }
                }
            });
        });
        Ok(())
    }
}

#[inline]
fn pixel_f64(px: &mut [u8], cr: f64, ci: f64, max_iter: i32) {
    let (mut zr, mut zi) = (0.0f64, 0.0f64);
    for n in 1..=max_iter {
        let (zr2, zi2) = (zr * zr, zi * zi);
        zi = 2.0 * zr * zi + ci;
        zr = zr2 - zi2 + cr;
        let r2 = zr * zr + zi * zi;
        if r2 > ESCAPE_R2 {
            return colour(px, n, r2);
        }
    }
    px.fill(0);
}

#[inline]
fn pixel_perturb(px: &mut [u8], r: &[f64], dcr: f64, dci: f64, max_iter: i32) {
    let ref_len = r.len() / 2;
    let (mut dzr, mut dzi) = (0.0f64, 0.0f64);
    let mut m = 0usize;
    for n in 1..=max_iter {
        // dz' = (2Z + dz) dz + dc
        let tr = 2.0 * r[2 * m] + dzr;
        let ti = 2.0 * r[2 * m + 1] + dzi;
        let nr = tr * dzr - ti * dzi + dcr;
        dzi = tr * dzi + ti * dzr + dci;
        dzr = nr;
        m += 1;
        let zr = r[2 * m] + dzr;
        let zi = r[2 * m + 1] + dzi;
        let r2 = zr * zr + zi * zi;
        if r2 > ESCAPE_R2 {
            return colour(px, n, r2);
        }
        if r2 < dzr * dzr + dzi * dzi || m == ref_len - 1 {
            dzr = zr; // rebase onto the start of the orbit
            dzi = zi;
            m = 0;
        }
    }
    px.fill(0);
}
