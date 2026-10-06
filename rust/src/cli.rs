use clap::{Parser, ValueEnum};

/// Default zoom centre: Misiurewicz point M(23,2), refined to 340 digits with
/// tools/misiurewicz.py. Boundary points like this have detail at every depth.
const DEFAULT_CENTRE_RE: &str = "-0.7766105925997018565640395025529947493281703214398660009692436578520468609045987087869614280320504142913293744385181191871713086676213649222522743166627219099561021923842471419613761736622513496509490647163308646781007447098756656445264983237348114769587849363429595360430698257987637989567458986520611392894629900349384819163368360786901345";
const DEFAULT_CENTRE_IM: &str = "0.1346089616750281660567372702330578095411874962204036173399792327554399692618243468637467375372086757134654215617641155866771648880044865998923176610765412471665957254547656437166294650322153138263014932943834549306648786205329836392513593309204091729193444399806460589188944478163945916072585799737463142109726240996892519451085765977598015";
const MAX_ZOOM: f64 = 1e300; // f64 deltas underflow beyond this

#[derive(Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum Device {
    Cpu,
    Gpu,
}

#[derive(Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum Precision {
    /// Plain doubles (zoom <= ~1e12).
    F64,
    /// Perturbation around a high-precision reference orbit (zoom <= 1e300).
    Deep,
}

impl Device {
    pub fn name(self) -> &'static str {
        match self {
            Device::Cpu => "cpu",
            Device::Gpu => "gpu",
        }
    }
}

impl Precision {
    pub fn name(self) -> &'static str {
        match self {
            Precision::F64 => "f64",
            Precision::Deep => "deep",
        }
    }
}

/// Renders a colour video zooming into the Mandelbrot set.
#[derive(Parser)]
#[command(name = "mandelbrot")]
pub struct Args {
    /// Compute device.
    #[arg(long, value_enum, default_value = "cpu")]
    pub device: Device,
    /// Arithmetic used for the iteration.
    #[arg(long, value_enum, default_value = "f64")]
    pub precision: Precision,
    #[arg(long, default_value_t = 1920)]
    pub width: i32,
    #[arg(long, default_value_t = 1080)]
    pub height: i32,
    #[arg(long, default_value_t = 30)]
    pub fps: i32,
    /// Video length in seconds.
    #[arg(long, default_value_t = 20.0)]
    pub duration: f64,
    /// Final magnification (default 1e12 for f64, 1e50 for deep).
    #[arg(long)]
    pub zoom: Option<f64>,
    /// Zoom centre, real part, as a decimal string.
    #[arg(long, default_value = DEFAULT_CENTRE_RE, hide_default_value = true)]
    pub centre_re: String,
    /// Zoom centre, imaginary part, as a decimal string.
    #[arg(long, default_value = DEFAULT_CENTRE_IM, hide_default_value = true)]
    pub centre_im: String,
    /// Maximum iterations at zoom 1.
    #[arg(long, default_value_t = 512)]
    pub iter_base: i32,
    /// Extra iterations per 10x zoom.
    #[arg(long, default_value_t = 128.0)]
    pub iter_per_decade: f64,
    /// CPU threads (0 = all).
    #[arg(long, default_value_t = 0)]
    pub threads: usize,
    /// ffmpeg video encoder.
    #[arg(long, default_value = "libx264")]
    pub encoder: String,
    /// Output file (default mandelbrot_rust_<device>_<precision>.mp4).
    #[arg(long)]
    pub output: Option<String>,
    /// Render only, skip ffmpeg (pure compute benchmark).
    #[arg(long)]
    pub no_video: bool,
}

impl Args {
    pub fn validated(mut self) -> anyhow::Result<Self> {
        anyhow::ensure!(
            self.width > 0 && self.height > 0 && self.fps > 0 && self.duration > 0.0 && self.iter_base > 0,
            "size, fps, duration and iter-base must be positive"
        );
        let zoom = *self.zoom.get_or_insert(match self.precision {
            Precision::F64 => 1e12,
            Precision::Deep => 1e50,
        });
        anyhow::ensure!((1.0..=MAX_ZOOM).contains(&zoom), "--zoom must be between 1 and 1e300");
        self.output.get_or_insert_with(|| {
            format!("mandelbrot_rust_{}_{}.mp4", self.device.name(), self.precision.name())
        });
        Ok(self)
    }
}
