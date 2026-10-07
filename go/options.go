package main

import (
	"flag"
	"fmt"
	"os"
)

// Default zoom centre: Misiurewicz point M(23,2), refined to 340 digits with
// tools/misiurewicz.py. Boundary points like this have detail at every depth.
const (
	defaultCentreRe = "-0.7766105925997018565640395025529947493281703214398660009692436578520468609045987087869614280320504142913293744385181191871713086676213649222522743166627219099561021923842471419613761736622513496509490647163308646781007447098756656445264983237348114769587849363429595360430698257987637989567458986520611392894629900349384819163368360786901345"
	defaultCentreIm = "0.1346089616750281660567372702330578095411874962204036173399792327554399692618243468637467375372086757134654215617641155866771648880044865998923176610765412471665957254547656437166294650322153138263014932943834549306648786205329836392513593309204091729193444399806460589188944478163945916072585799737463142109726240996892519451085765977598015"
	maxZoom         = 1e300 // f64 deltas underflow beyond this
)

// Options are the command-line settings, shared with the C++ and Rust versions.
type Options struct {
	Device, Precision  string
	Width, Height, FPS int
	Duration, Zoom     float64
	CentreRe, CentreIm string
	IterBase           int
	IterPerDecade      float64
	Threads            int
	Encoder, Output    string
	NoVideo            bool
	GPUFP32            bool
}

func parseOptions() Options {
	var o Options
	flag.StringVar(&o.Device, "device", "cpu", "compute device: cpu|gpu|both (both: CPU and GPU render alternate frames)")
	flag.StringVar(&o.Precision, "precision", "f64", "f64 = plain doubles (zoom <= ~1e12), deep = perturbation (zoom <= 1e300)")
	flag.IntVar(&o.Width, "width", 1920, "frame width")
	flag.IntVar(&o.Height, "height", 1080, "frame height")
	flag.IntVar(&o.FPS, "fps", 30, "frames per second")
	flag.Float64Var(&o.Duration, "duration", 20, "video length in seconds")
	flag.Float64Var(&o.Zoom, "zoom", 0, "final magnification (default 1e12 f64, 1e50 deep)")
	flag.StringVar(&o.CentreRe, "centre-re", defaultCentreRe, "zoom centre, real part, as a decimal string")
	flag.StringVar(&o.CentreIm, "centre-im", defaultCentreIm, "zoom centre, imaginary part, as a decimal string")
	flag.IntVar(&o.IterBase, "iter-base", 512, "max iterations at zoom 1")
	flag.Float64Var(&o.IterPerDecade, "iter-per-decade", 128, "extra iterations per 10x zoom")
	flag.IntVar(&o.Threads, "threads", 0, "CPU threads (0 = all)")
	flag.StringVar(&o.Encoder, "encoder", "libx264", "ffmpeg video encoder")
	flag.StringVar(&o.Output, "output", "", "output file (default mandelbrot_go_<device>_<precision>.mp4)")
	flag.BoolVar(&o.NoVideo, "no-video", false, "render only, skip ffmpeg (pure compute benchmark)")
	flag.BoolVar(&o.GPUFP32, "gpu-fp32", false, "deep mode on the GPU in float32: much faster, not bit-identical to the CPU (zoom <= ~1e63)")
	flag.Parse()

	fail := func(msg string) {
		fmt.Fprintln(os.Stderr, "error:", msg)
		flag.Usage()
		os.Exit(2)
	}
	if o.Device != "cpu" && o.Device != "gpu" && o.Device != "both" {
		fail("--device must be cpu, gpu or both")
	}
	if o.Precision != "f64" && o.Precision != "deep" {
		fail("--precision must be f64 or deep")
	}
	if o.Width <= 0 || o.Height <= 0 || o.FPS <= 0 || o.Duration <= 0 || o.IterBase <= 0 {
		fail("size, fps, duration and iter-base must be positive")
	}
	if o.Zoom == 0 {
		o.Zoom = map[string]float64{"f64": 1e12, "deep": 1e50}[o.Precision]
	}
	if o.Zoom < 1 || o.Zoom > maxZoom {
		fail("--zoom must be between 1 and 1e300")
	}
	if o.GPUFP32 && (o.Precision != "deep" || o.Device == "cpu") {
		fail("--gpu-fp32 needs --precision deep and --device gpu or both")
	}
	if o.Output == "" {
		o.Output = fmt.Sprintf("mandelbrot_go_%s_%s.mp4", o.Device, o.Precision)
	}
	return o
}
