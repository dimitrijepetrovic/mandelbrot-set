// Command mandelbrot renders a colour video zooming into the Mandelbrot set.
package main

import (
	"errors"
	"fmt"
	"math"
	"os"
	"strconv"
	"sync"
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

	var renderers []Renderer
	if o.Device != "gpu" {
		renderers = append(renderers, newCPURenderer(o, cre, cim, orbit))
	}
	if o.Device != "cpu" {
		g, err := newGPURenderer(o, cre, cim, orbit)
		if err != nil {
			return err
		}
		defer g.Close()
		renderers = append(renderers, g)
	}

	var video *VideoWriter
	if !o.NoVideo {
		if video, err = NewVideoWriter(o.Output, o.Width, o.Height, o.FPS, o.Encoder); err != nil {
			return err
		}
	}

	// One worker goroutine per device takes the next frame number from a shared counter,
	// so with --device both the faster device renders more frames. Near the end a device
	// stops if the other would finish all remaining frames before it finished one more
	// (the last frames are the most expensive), so neither waits on the other. This goroutine writes
	// finished frames to ffmpeg in order while the workers render ahead, so encoding
	// overlaps rendering. Frame buffers are allocated as needed, so a fast device can
	// run ahead of a slow one's frame (up to maxBufs frames in flight).
	type result struct {
		i   int
		rgb []byte
	}
	frameBytes := o.Width * o.Height * 3
	maxBufs := min(max((1<<30)/frameBytes, 3), 64)
	free := make(chan []byte, maxBufs) // buffers to render into
	done := make(chan result, maxBufs)
	stop := make(chan struct{}) // closed when anything fails
	var stopOnce sync.Once
	abort := func() { stopOnce.Do(func() { close(stop) }) }
	errs := make([]error, len(renderers))
	rendered := make([]int, len(renderers))
	var mu sync.Mutex // guards nextFrame, lastS and bufs
	nextFrame, lastS, bufs := 0, make([]float64, len(renderers)), 0
	newBuf := func() []byte { // a new buffer, or nil at maxBufs
		mu.Lock()
		defer mu.Unlock()
		if bufs == maxBufs {
			return nil
		}
		bufs++
		return make([]byte, frameBytes)
	}
	claim := func(w int) (int, bool) { // the next frame for worker w, or false to stop
		mu.Lock()
		defer mu.Unlock()
		for v := range lastS {
			if v != w && lastS[v] > 0 && float64(frames-nextFrame)*lastS[v] < lastS[w] {
				return 0, false
			}
		}
		nextFrame++
		return nextFrame - 1, nextFrame <= frames
	}
	var wg sync.WaitGroup
	renderStart := time.Now()
	for w, r := range renderers {
		wg.Go(func() {
			for {
				var rgb []byte
				select {
				case rgb = <-free:
				default:
					if rgb = newBuf(); rgb == nil {
						select {
						case rgb = <-free:
						case <-stop:
							return
						}
					}
				}
				i, ok := claim(w)
				if !ok {
					return
				}
				t0 := time.Now()
				if errs[w] = r.Render(frameAt(i), rgb); errs[w] != nil {
					abort()
					return
				}
				mu.Lock()
				lastS[w] = time.Since(t0).Seconds()
				mu.Unlock()
				rendered[w]++
				done <- result{i, rgb} // never blocks: done holds every buffer
			}
		})
	}
	writeErr := func() error {
		pending := map[int][]byte{} // rendered, not yet written
		for i := 0; i < frames; {
			rgb, ok := pending[i]
			if !ok {
				select {
				case res := <-done:
					pending[res.i] = res.rgb
				case <-stop:
					return nil // a worker's error is reported below
				}
				continue
			}
			delete(pending, i)
			if video != nil {
				if err := video.Write(rgb); err != nil {
					return err
				}
			}
			fmt.Fprintf(os.Stderr, "\rframe %d/%d  max_iter %d", i+1, frames, frameAt(i).MaxIter)
			free <- rgb
			i++
		}
		return nil
	}()
	if writeErr != nil {
		abort()
	}
	wg.Wait()
	fmt.Fprintln(os.Stderr)
	if err := errors.Join(append(errs, writeErr)...); err != nil {
		return err
	}
	// renderS: until the last frame was rendered and written (encoding overlaps it);
	// writeS: flushing ffmpeg after that.
	renderS, writeS := time.Since(renderStart).Seconds(), 0.0
	output := "-"
	if video != nil {
		t0 := time.Now()
		if err := video.Finish(); err != nil {
			return err
		}
		writeS = time.Since(t0).Seconds()
		output = o.Output
	}
	if o.Device == "both" {
		fmt.Fprintf(os.Stderr, "frames rendered: cpu %d, gpu %d\n", rendered[0], rendered[1])
	}
	totalS := time.Since(start).Seconds()

	fmt.Printf("lang=go device=%s precision=%s size=%dx%d frames=%d zoom=%.3g ref_s=%.3f "+
		"render_s=%.3f write_s=%.3f total_s=%.3f render_fps=%.2f output=%s\n",
		o.Device, precisionName(o), o.Width, o.Height, frames, o.Zoom, refS,
		renderS, writeS, totalS, float64(frames)/renderS, output)
	return nil
}

// precisionName is the summary line's precision: deep-fp32 for --gpu-fp32.
func precisionName(o Options) string {
	if o.GPUFP32 {
		return "deep-fp32"
	}
	return o.Precision
}
