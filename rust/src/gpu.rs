use std::sync::Arc;

use cudarc::driver::{CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::Ptx;

use crate::cli::Args;
use crate::{Frame, Renderer};

const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/mandelbrot.ptx"));

/// 2^-220, MANDEL_F32_MIN_SPACING in cuda/mandelbrot.cuh: below this spacing the float32
/// kernel's scaled deltas would overflow, so deeper frames use the float64 kernel.
const F32_MIN_SPACING: f64 = 5.9241243523004765e-67;

/// The reference orbit on the GPU, in float64 and (with --gpu-fp32) rounded to float32.
struct Orbit {
    f64: CudaSlice<f64>,
    f32: Option<CudaSlice<f32>>,
    len: i32,
}

pub struct GpuRenderer {
    width: i32,
    height: i32,
    cre: f64,
    cim: f64,
    stream: Arc<CudaStream>,
    mandel_f64: CudaFunction,
    mandel_perturb: CudaFunction,
    mandel_perturb_f32: CudaFunction,
    rgb: CudaSlice<u8>,
    orbit: Option<Orbit>,
}

impl GpuRenderer {
    pub fn new(args: &Args, cre: f64, cim: f64, orbit: Option<Vec<f64>>) -> anyhow::Result<Self> {
        let ctx = cudarc::driver::CudaContext::new(0)?;
        let stream = ctx.default_stream();
        let module = ctx.load_module(Ptx::from_src(PTX))?;
        let rgb = stream.alloc_zeros::<u8>(args.width as usize * args.height as usize * 3)?;
        let orbit = match orbit {
            Some(o) => Some(Orbit {
                f64: stream.clone_htod(&o)?,
                f32: if args.gpu_fp32 {
                    Some(stream.clone_htod(&o.iter().map(|&v| v as f32).collect::<Vec<_>>())?)
                } else {
                    None
                },
                len: (o.len() / 2) as i32,
            }),
            None => None,
        };
        Ok(Self {
            width: args.width,
            height: args.height,
            cre,
            cim,
            mandel_f64: module.load_function("mandel_f64")?,
            mandel_perturb: module.load_function("mandel_perturb")?,
            mandel_perturb_f32: module.load_function("mandel_perturb_f32")?,
            stream,
            rgb,
            orbit,
        })
    }
}

impl Renderer for GpuRenderer {
    fn render(&mut self, f: Frame, rgb: &mut [u8]) -> anyhow::Result<()> {
        let cfg = LaunchConfig {
            grid_dim: ((self.width as u32).div_ceil(16), (self.height as u32).div_ceil(16), 1),
            block_dim: (16, 16, 1),
            shared_mem_bytes: 0,
        };
        let kernel = match &self.orbit {
            Some(Orbit { f32: Some(_), .. }) if f.spacing >= F32_MIN_SPACING => &self.mandel_perturb_f32,
            Some(_) => &self.mandel_perturb,
            None => &self.mandel_f64,
        };
        let mut launch = self.stream.launch_builder(kernel);
        launch.arg(&mut self.rgb).arg(&self.width).arg(&self.height);
        match &self.orbit {
            Some(o @ Orbit { f32: Some(ref32), .. }) if f.spacing >= F32_MIN_SPACING => launch.arg(ref32).arg(&o.len),
            Some(o) => launch.arg(&o.f64).arg(&o.len),
            None => launch.arg(&self.cre).arg(&self.cim),
        };
        launch.arg(&f.spacing).arg(&f.max_iter);
        // SAFETY: argument types and order match the kernel signatures in cuda/mandelbrot.cuh.
        unsafe { launch.launch(cfg) }?;
        self.stream.memcpy_dtoh(&self.rgb, rgb)?;
        Ok(())
    }
}
