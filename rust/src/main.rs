mod cli;
mod cpu;
mod gpu;
mod reference;
mod video;

use std::collections::BTreeMap;
use std::io::Write;
use std::sync::{Condvar, Mutex};
use std::thread;
use std::time::Instant;

use clap::Parser;
use cli::{Args, Device, Precision};

// Must match cuda/mandelbrot.cu and the C++/Go implementations.
pub const ESCAPE_R2: f64 = 65536.0;
const COLOUR_DENSITY: f64 = 0.015;
const TWO_PI: f64 = 6.283185307179586;
pub const LN2: f64 = 0.6931471805599453;
const LN10: f64 = 2.302585092994046;
const SQRT_HALF: f64 = 0.7071067811865476;
const VIEW_HEIGHT: f64 = 3.0; // complex-plane height of frame 0

/// Per-frame view parameters.
#[derive(Clone, Copy)]
pub struct Frame {
    /// Complex-plane distance between adjacent pixels.
    pub spacing: f64,
    pub max_iter: i32,
}

/// Frames shared between the render workers and the writer (see `run`).
struct Pipeline {
    free: Vec<Vec<u8>>,           // buffers to render into
    bufs: usize,                  // buffers allocated
    done: BTreeMap<i32, Vec<u8>>, // rendered, not yet written
    rendered: Vec<i32>,           // frames per worker
    last_s: Vec<f64>,             // each worker's latest frame time
    next_frame: i32,              // the next frame to render
    failed: bool,                 // a worker or the writer failed: stop
}

impl Pipeline {
    /// Whether worker `w` should stop: another worker would render all remaining frames
    /// before `w` finished one more.
    fn yields(&self, w: usize, frames: i32) -> bool {
        let remaining = (frames - self.next_frame) as f64;
        (0..self.last_s.len()).any(|v| v != w && self.last_s[v] > 0.0 && remaining * self.last_s[v] < self.last_s[w])
    }
}

pub trait Renderer: Send {
    fn render(&mut self, frame: Frame, rgb: &mut [u8]) -> anyhow::Result<()>;
}

#[inline]
pub fn colour(px: &mut [u8], n: i32, r2: f64) {
    let log_zn = 0.5 * r2.ln();
    let nu = n as f64 + 1.0 - (log_zn / LN2).ln() / LN2;
    let t = nu * COLOUR_DENSITY;
    px[0] = ((0.5 + 0.5 * (TWO_PI * (t + 0.00)).cos()) * 255.0 + 0.5) as u8;
    px[1] = ((0.5 + 0.5 * (TWO_PI * (t + 0.15)).cos()) * 255.0 + 0.5) as u8;
    px[2] = ((0.5 + 0.5 * (TWO_PI * (t + 0.30)).cos()) * 255.0 + 0.5) as u8;
}

/// exp/ln built only from correctly rounded IEEE operations and exact power-of-two
/// scaling, so all three languages compute bit-identical frame parameters
/// (libm pow/log/exp differ in the last bit between Go and C).
fn det_exp(x: f64) -> f64 {
    let k = (x / LN2).round();
    let r = x - k * LN2;
    let mut s = 1.0;
    for i in (1..=24).rev() {
        s = 1.0 + s * r / i as f64;
    }
    s * f64::from_bits(((k as i64 + 1023) as u64) << 52) // exact 2^k
}

pub fn det_ln(x: f64) -> f64 {
    // x = m * 2^e with m in [0.5, 1); x is a positive normal number here.
    let bits = x.to_bits();
    let mut e = ((bits >> 52) & 0x7ff) as i32 - 1022;
    let mut m = f64::from_bits((bits & !(0x7ff << 52)) | (1022 << 52));
    if m < SQRT_HALF {
        m *= 2.0;
        e -= 1;
    }
    let s = (m - 1.0) / (m + 1.0);
    let s2 = s * s;
    let mut sum = 0.0;
    for i in (1..=41).rev().step_by(2) {
        sum = sum * s2 + 1.0 / i as f64;
    }
    2.0 * s * sum + e as f64 * LN2
}

fn main() {
    if let Err(e) = run() {
        eprintln!("\nerror: {e:#}");
        std::process::exit(1);
    }
}

fn run() -> anyhow::Result<()> {
    let args = Args::parse().validated()?;
    let deep = args.precision == Precision::Deep;
    let zoom = args.zoom.unwrap();
    let frames = ((args.fps as f64 * args.duration).round() as i32).max(1);
    let ln_zoom = det_ln(zoom);

    // Frame i zooms exponentially from 1x to `zoom`; iterations grow with depth.
    let frame_at = |i: i32| {
        let t = if frames > 1 { i as f64 / (frames - 1) as f64 } else { 0.0 };
        Frame {
            spacing: VIEW_HEIGHT / (det_exp(t * ln_zoom) * args.height as f64),
            max_iter: args.iter_base + (args.iter_per_decade * t * ln_zoom / LN10) as i32,
        }
    };

    let start = Instant::now();
    let mut ref_s = 0.0;
    let orbit = if deep {
        let t0 = Instant::now();
        let o = reference::orbit(&args.centre_re, &args.centre_im, frame_at(frames - 1).max_iter, ln_zoom)?;
        ref_s = t0.elapsed().as_secs_f64();
        Some(o)
    } else {
        None
    };
    let cre: f64 = args.centre_re.parse()?;
    let cim: f64 = args.centre_im.parse()?;
    let mut renderers: Vec<Box<dyn Renderer>> = Vec::new();
    if args.device != Device::Gpu {
        renderers.push(Box::new(cpu::CpuRenderer::new(&args, cre, cim, orbit.clone())?));
    }
    if args.device != Device::Cpu {
        renderers.push(Box::new(gpu::GpuRenderer::new(&args, cre, cim, orbit)?));
    }

    let output = args.output.clone().unwrap();
    let mut video = if args.no_video {
        None
    } else {
        Some(video::VideoWriter::new(&output, args.width, args.height, args.fps, &args.encoder)?)
    };

    // One worker thread per device takes the next frame number from a shared counter,
    // so with --device both the faster device renders more frames. Near the end a device
    // stops if the other would finish all remaining frames before it finished one more
    // (the last frames are the most expensive), so neither waits on the other. The main thread
    // writes finished frames to ffmpeg in order while the workers render ahead, so
    // encoding overlaps rendering. Frame buffers are allocated as needed, so a fast
    // device can run ahead of a slow one's frame (up to max_bufs frames in flight).
    let frame_bytes = args.width as usize * args.height as usize * 3;
    let max_bufs = ((1usize << 30) / frame_bytes).clamp(3, 64);
    let state = Mutex::new(Pipeline {
        free: Vec::new(),
        bufs: 0,
        done: BTreeMap::new(),
        rendered: vec![0; renderers.len()],
        last_s: vec![0.0; renderers.len()],
        next_frame: 0,
        failed: false,
    });
    let cv = Condvar::new();
    let render_start = Instant::now();
    let result = thread::scope(|scope| -> anyhow::Result<()> {
        let workers: Vec<_> = renderers
            .iter_mut()
            .enumerate()
            .map(|(w, r)| {
                let (state, cv) = (&state, &cv);
                scope.spawn(move || -> anyhow::Result<()> {
                    let result = (|| loop {
                        let (i, mut rgb) = {
                            let mut s = cv
                                .wait_while(state.lock().unwrap(), |s| s.free.is_empty() && s.bufs >= max_bufs && !s.failed)
                                .unwrap();
                            if s.failed || s.next_frame >= frames || s.yields(w, frames) {
                                return Ok(());
                            }
                            s.next_frame += 1;
                            let rgb = s.free.pop().unwrap_or_else(|| {
                                s.bufs += 1;
                                vec![0u8; frame_bytes]
                            });
                            (s.next_frame - 1, rgb)
                        };
                        let t0 = Instant::now();
                        r.render(frame_at(i), &mut rgb)?;
                        let mut s = state.lock().unwrap();
                        s.last_s[w] = t0.elapsed().as_secs_f64();
                        s.done.insert(i, rgb);
                        s.rendered[w] += 1;
                        cv.notify_all();
                    })();
                    if result.is_err() {
                        state.lock().unwrap().failed = true;
                        cv.notify_all();
                    }
                    result
                })
            })
            .collect();
        let written = (|| -> anyhow::Result<()> {
            for i in 0..frames {
                let rgb = {
                    let mut s = cv.wait_while(state.lock().unwrap(), |s| !s.done.contains_key(&i) && !s.failed).unwrap();
                    if s.failed {
                        return Ok(()); // a worker's error is reported below
                    }
                    s.done.remove(&i).unwrap()
                };
                if let Some(v) = video.as_mut() {
                    v.write(&rgb)?;
                }
                eprint!("\rframe {}/{}  max_iter {}", i + 1, frames, frame_at(i).max_iter);
                std::io::stderr().flush().ok();
                state.lock().unwrap().free.push(rgb);
                cv.notify_all();
            }
            Ok(())
        })();
        if written.is_err() {
            state.lock().unwrap().failed = true;
            cv.notify_all();
        }
        for w in workers {
            w.join().unwrap()?;
        }
        written
    });
    eprintln!();
    result?;
    // render_s: until the last frame was rendered and written (encoding overlaps it);
    // write_s: flushing ffmpeg after that.
    let render_s = render_start.elapsed().as_secs_f64();
    let mut write_s = 0.0;
    if let Some(v) = video {
        let t0 = Instant::now();
        v.finish()?;
        write_s = t0.elapsed().as_secs_f64();
    }
    if args.device == Device::Both {
        let r = &state.lock().unwrap().rendered;
        eprintln!("frames rendered: cpu {}, gpu {}", r[0], r[1]);
    }
    let total_s = start.elapsed().as_secs_f64();

    println!(
        "lang=rust device={} precision={} size={}x{} frames={} zoom={} ref_s={:.3} render_s={:.3} \
         write_s={:.3} total_s={:.3} render_fps={:.2} output={}",
        args.device.name(),
        if args.gpu_fp32 { "deep-fp32" } else { args.precision.name() },
        args.width,
        args.height,
        frames,
        format_zoom(zoom),
        ref_s,
        render_s,
        write_s,
        total_s,
        frames as f64 / render_s,
        if args.no_video { "-" } else { &output },
    );
    Ok(())
}

/// Formats like C's "%.3g" so summary lines match across languages.
fn format_zoom(z: f64) -> String {
    let s = format!("{z:.2e}");
    let (m, e) = s.split_once('e').unwrap();
    let m = m.trim_end_matches('0').trim_end_matches('.');
    let e: i32 = e.parse().unwrap();
    format!("{m}e{}{:02}", if e < 0 { '-' } else { '+' }, e.abs())
}
