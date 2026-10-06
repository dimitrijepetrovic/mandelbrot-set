package main

import (
	"runtime"
	"sync"
	"sync/atomic"
)

type cpuRenderer struct {
	width, height, threads int
	cre, cim               float64
	orbit                  []float64
}

func newCPURenderer(o Options, cre, cim float64, orbit []float64) *cpuRenderer {
	threads := o.Threads
	if threads <= 0 {
		threads = runtime.NumCPU()
	}
	return &cpuRenderer{o.Width, o.Height, threads, cre, cim, orbit}
}

func (r *cpuRenderer) Render(f Frame, rgb []byte) error {
	// Rows are handed out dynamically: escape times vary a lot across the image.
	var nextRow atomic.Int64
	var wg sync.WaitGroup
	for range r.threads {
		wg.Go(func() {
			for y := int(nextRow.Add(1) - 1); y < r.height; y = int(nextRow.Add(1) - 1) {
				row := rgb[y*r.width*3 : (y+1)*r.width*3]
				dci := (0.5*float64(r.height) - (float64(y) + 0.5)) * f.Spacing
				for x := range r.width {
					dcr := (float64(x) + 0.5 - 0.5*float64(r.width)) * f.Spacing
					px := row[3*x : 3*x+3]
					if r.orbit != nil {
						pixelPerturb(px, r.orbit, dcr, dci, f.MaxIter)
					} else {
						pixelF64(px, r.cre+dcr, r.cim+dci, f.MaxIter)
					}
				}
			}
		})
	}
	wg.Wait()
	return nil
}

func pixelF64(px []byte, cr, ci float64, maxIter int32) {
	var zr, zi float64
	for n := int32(1); n <= maxIter; n++ {
		zr2, zi2 := zr*zr, zi*zi
		zi = 2.0*zr*zi + ci
		zr = zr2 - zi2 + cr
		r2 := zr*zr + zi*zi
		if r2 > escapeR2 {
			colour(px, n, r2)
			return
		}
	}
	px[0], px[1], px[2] = 0, 0, 0
}

func pixelPerturb(px []byte, ref []float64, dcr, dci float64, maxIter int32) {
	refLen := len(ref) / 2
	var dzr, dzi float64
	m := 0
	for n := int32(1); n <= maxIter; n++ {
		// dz' = (2Z + dz) dz + dc
		tr := 2.0*ref[2*m] + dzr
		ti := 2.0*ref[2*m+1] + dzi
		nr := tr*dzr - ti*dzi + dcr
		dzi = tr*dzi + ti*dzr + dci
		dzr = nr
		m++
		zr := ref[2*m] + dzr
		zi := ref[2*m+1] + dzi
		r2 := zr*zr + zi*zi
		if r2 > escapeR2 {
			colour(px, n, r2)
			return
		}
		if r2 < dzr*dzr+dzi*dzi || m == refLen-1 {
			dzr, dzi, m = zr, zi, 0 // rebase onto the start of the orbit
		}
	}
	px[0], px[1], px[2] = 0, 0, 0
}
