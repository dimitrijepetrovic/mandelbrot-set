// Command mandelbrot renders a colour video zooming into the Mandelbrot set.
package main

import (
	"fmt"
	"math"
	"os"
	"strconv"
	"time"
)

// Must match cuda/mandelbrot.cu and the C++/Rust implementations.
const (
	escapeR2      = 65536.0
	colourDensity = 0.015
	twoPi         = 6.283185307179586
	ln2           = 0.6931471805599453
	ln10          = 2.302585092994046
	sqrtHalf      = 0.7071067811865476
	viewHeight    = 3.0 // complex-plane height of frame 0
)

// Frame holds the per-frame view parameters.
type Frame struct {
	Spacing float64 // complex-plane distance between adjacent pixels
	MaxIter int32
}

// Renderer fills rgb (width*height*3 bytes) with one frame.
type Renderer interface {
	Render(f Frame, rgb []byte) error
}

func colour(px []byte, n int32, r2 float64) {
	logZn := 0.5 * math.Log(r2)
	nu := float64(n) + 1.0 - math.Log(logZn/ln2)/ln2
	t := nu * colourDensity
	px[0] = byte((0.5+0.5*math.Cos(twoPi*(t+0.00)))*255.0 + 0.5)
	px[1] = byte((0.5+0.5*math.Cos(twoPi*(t+0.15)))*255.0 + 0.5)
	px[2] = byte((0.5+0.5*math.Cos(twoPi*(t+0.30)))*255.0 + 0.5)
}

// detExp and detLn are built only from correctly rounded IEEE operations and
// exact power-of-two scaling, so all three languages compute bit-identical
// frame parameters (Go's math.Pow/Log differ from C's libm in the last bit).
func detExp(x float64) float64 {
	k := math.Round(x / ln2)
	r := x - k*ln2
	s := 1.0
	for i := 24; i >= 1; i-- {
		s = 1.0 + s*r/float64(i)
	}
	return math.Ldexp(s, int(k))
}

func detLn(x float64) float64 {
	m, e := math.Frexp(x) // x = m * 2^e, m in [0.5, 1)
	if m < sqrtHalf {
		m *= 2.0
		e--
	}
	s := (m - 1.0) / (m + 1.0)
	s2 := s * s
	sum := 0.0
	for i := 41; i >= 1; i -= 2 {
		sum = sum*s2 + 1.0/float64(i)
	}
	return 2.0*s*sum + float64(e)*ln2
}

func main() {
	if err := run(); err != nil {
		fmt.Fprintf(os.Stderr, "\nerror: %v\n", err)
		os.Exit(1)
	}
}

func run() error {
	o := parseOptions()
	frames := max(1, int(math.Round(float64(o.FPS)*o.Duration)))
	lnZoom := detLn(o.Zoom)

	// Frame i zooms exponentially from 1x to o.Zoom; iterations grow with depth.
	frameAt := func(i int) Frame {
		t := 0.0
		if frames > 1 {
			t = float64(i) / float64(frames-1)
		}
		return Frame{
			Spacing: viewHeight / (detExp(t*lnZoom) * float64(o.Height)),
			MaxIter: int32(o.IterBase) + int32(o.IterPerDecade*t*lnZoom/ln10),
		}
	}

	start := time.Now()
	refS := 0.0
	var orbit []float64
	if o.Precision == "deep" {
		t0 := time.Now()
		var err error
		if orbit, err = referenceOrbit(o.CentreRe, o.CentreIm, frameAt(frames-1).MaxIter, lnZoom); err != nil {
			return err
		}
		refS = time.Since(t0).Seconds()
	}
	cre, err := strconv.ParseFloat(o.CentreRe, 64)
	if err != nil {
		return fmt.Errorf("invalid centre: %w", err)
	}
	cim, err := strconv.ParseFloat(o.CentreIm, 64)
	if err != nil {
		return fmt.Errorf("invalid centre: %w", err)
	}

	var renderer Renderer
	if o.Device == "cpu" {
		renderer = newCPURenderer(o, cre, cim, orbit)
	} else {
		g, err := newGPURenderer(o, cre, cim, orbit)
		if err != nil {
			return err
		}
		defer g.Close()
		renderer = g
	}

	var video *VideoWriter
	if !o.NoVideo {
		if video, err = NewVideoWriter(o.Output, o.Width, o.Height, o.FPS, o.Encoder); err != nil {
			return err
		}
	}

	rgb := make([]byte, o.Width*o.Height*3)
	var renderS, writeS float64
	for i := range frames {
		f := frameAt(i)
		t0 := time.Now()
		if err := renderer.Render(f, rgb); err != nil {
			return err
		}
		renderS += time.Since(t0).Seconds()
		if video != nil {
			t0 = time.Now()
			if err := video.Write(rgb); err != nil {
				return err
			}
			writeS += time.Since(t0).Seconds()
		}
		fmt.Fprintf(os.Stderr, "\rframe %d/%d  max_iter %d", i+1, frames, f.MaxIter)
	}
	fmt.Fprintln(os.Stderr)
	output := "-"
	if video != nil {
		t0 := time.Now()
		if err := video.Finish(); err != nil {
			return err
		}
		writeS += time.Since(t0).Seconds()
		output = o.Output
	}
	totalS := time.Since(start).Seconds()

	fmt.Printf("lang=go device=%s precision=%s size=%dx%d frames=%d zoom=%.3g ref_s=%.3f "+
		"render_s=%.3f write_s=%.3f total_s=%.3f render_fps=%.2f output=%s\n",
		o.Device, o.Precision, o.Width, o.Height, frames, o.Zoom, refS,
		renderS, writeS, totalS, float64(frames)/renderS, output)
	return nil
}
