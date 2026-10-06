use std::sync::Arc;

use cudarc::driver::{CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::Ptx;

use crate::cli::Args;
use crate::{Frame, Renderer};

const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/mandelbrot.ptx"));

pub struct GpuRenderer {
    width: i32,
    height: i32,
    cre: f64,
    cim: f64,
    stream: Arc<CudaStream>,
    kernel: CudaFunction,
    rgb: CudaSlice<u8>,
    orbit: Option<(CudaSlice<f64>, i32)>,
}

impl GpuRenderer {
    pub fn new(args: &Args, cre: f64, cim: f64, orbit: Option<Vec<f64>>) -> anyhow::Result<Self> {
        let ctx = cudarc::driver::CudaContext::new(0)?;
        let stream = ctx.default_stream();
        let module = ctx.load_module(Ptx::from_src(PTX))?;
        let kernel = module.load_function(if orbit.is_some() { "mandel_perturb" } else { "mandel_f64" })?;
        let rgb = stream.alloc_zeros::<u8>(args.width as usize * args.height as usize * 3)?;
        let orbit = match orbit {
            Some(o) => Some((stream.clone_htod(&o)?, (o.len() / 2) as i32)),
            None => None,
        };
        Ok(Self { width: args.width, height: args.height, cre, cim, stream, kernel, rgb, orbit })
    }
}

impl Renderer for GpuRenderer {
    fn render(&mut self, f: Frame, rgb: &mut [u8]) -> anyhow::Result<()> {
        let cfg = LaunchConfig {
            grid_dim: ((self.width as u32).div_ceil(16), (self.height as u32).div_ceil(16), 1),
            block_dim: (16, 16, 1),
            shared_mem_bytes: 0,
        };
        let mut launch = self.stream.launch_builder(&self.kernel);
        launch.arg(&mut self.rgb).arg(&self.width).arg(&self.height);
        match &self.orbit {
            Some((orbit, len)) => launch.arg(orbit).arg(len),
            None => launch.arg(&self.cre).arg(&self.cim),
        };
        launch.arg(&f.spacing).arg(&f.max_iter);
        // SAFETY: argument types and order match the kernel signatures in cuda/mandelbrot.cuh.
        unsafe { launch.launch(cfg) }?;
        self.stream.memcpy_dtoh(&self.rgb, rgb)?;
        Ok(())
    }
}
