mod cli;
mod cpu;
mod gpu;
mod reference;
mod video;

use std::io::Write;
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

pub trait Renderer {
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
    let mut renderer: Box<dyn Renderer> = match args.device {
        Device::Cpu => Box::new(cpu::CpuRenderer::new(&args, cre, cim, orbit)?),
        Device::Gpu => Box::new(gpu::GpuRenderer::new(&args, cre, cim, orbit)?),
    };

    let output = args.output.clone().unwrap();
    let mut video = if args.no_video {
        None
    } else {
        Some(video::VideoWriter::new(&output, args.width, args.height, args.fps, &args.encoder)?)
    };

    let mut rgb = vec![0u8; args.width as usize * args.height as usize * 3];
    let (mut render_s, mut write_s) = (0.0, 0.0);
    for i in 0..frames {
        let f = frame_at(i);
        let t0 = Instant::now();
        renderer.render(f, &mut rgb)?;
        render_s += t0.elapsed().as_secs_f64();
        if let Some(v) = video.as_mut() {
            let t0 = Instant::now();
            v.write(&rgb)?;
            write_s += t0.elapsed().as_secs_f64();
        }
        eprint!("\rframe {}/{}  max_iter {}", i + 1, frames, f.max_iter);
        std::io::stderr().flush().ok();
    }
    eprintln!();
    if let Some(v) = video {
        let t0 = Instant::now();
        v.finish()?;
        write_s += t0.elapsed().as_secs_f64();
    }
    let total_s = start.elapsed().as_secs_f64();

    println!(
        "lang=rust device={} precision={} size={}x{} frames={} zoom={} ref_s={:.3} render_s={:.3} \
         write_s={:.3} total_s={:.3} render_fps={:.2} output={}",
        args.device.name(),
        args.precision.name(),
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
