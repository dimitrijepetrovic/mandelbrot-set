//go:build goexperiment.simd

package main

import (
	"math"
	"simd/archsimd"
)

// Each worker iterates 8 pixels of a row at once, as two AVX2 vectors of 4 lanes. One
// pixel's iterations form a serial dependency chain (each step needs the previous z),
// so a single pixel leaves the core mostly idle; independent lanes fill it. When a lane
// finishes, it is coloured and refilled with the row's next pixel. Every lane does
// exactly the scalar operations of the one-pixel loop (no FMA), so the output is
// unchanged. Go doesn't auto-vectorize, so this uses the experimental simd/archsimd
// package (GOEXPERIMENT=simd), where the C++ and Rust compilers vectorize plain loops.

const lanes = 8

const idle = -1

var hasAVX2 = archsimd.X86.AVX2()

type f64x4 = archsimd.Float64x4

// Lane state lives in [lanes]float64 arrays between refills; the hot loops keep it in
// two 4-wide vectors per quantity (lo = lanes 0-3, hi = lanes 4-7). Separate variables,
// not arrays of vectors, so the compiler keeps them in registers.
func load(a *[lanes]float64) (lo, hi f64x4) {
	return archsimd.LoadFloat64x4((*[4]float64)(a[0:4])), archsimd.LoadFloat64x4((*[4]float64)(a[4:8]))
}

func store(a *[lanes]float64, lo, hi f64x4) {
	lo.Store((*[4]float64)(a[0:4]))
	hi.Store((*[4]float64)(a[4:8]))
}

// feed hands out a row's pixels to lanes.
type feed struct{ next, width, active int }

// take returns the next pixel's index, or idle when the row is used up.
func (f *feed) take() int {
	if f.next >= f.width {
		return idle
	}
	f.next++
	f.active++
	return f.next - 1
}

// finish writes the colour of a finished lane: escaped, or black at max_iter.
func finish(row []byte, px int, n, r2 float64) {
	p := row[3*px : 3*px+3]
	if r2 > escapeR2 {
		colour(p, int32(n), r2)
	} else {
		p[0], p[1], p[2] = 0, 0, 0
	}
}

// stepF64 is one iteration of z <- z^2 + c for 4 lanes; it returns the new z and |z|^2.
func stepF64(zr, zi, cr, ci, two f64x4) (f64x4, f64x4, f64x4) {
	zr2, zi2 := zr.Mul(zr), zi.Mul(zi)
	ni := two.Mul(zr).Mul(zi).Add(ci)
	nr := zr2.Sub(zi2).Add(cr)
	return nr, ni, nr.Mul(nr).Add(ni.Mul(ni))
}

// rowF64 iterates z <- z^2 + c from z = 0.
func rowF64(row []byte, width int, cre, ciRow, spacing float64, maxIter int32) {
	if !hasAVX2 {
		rowF64Scalar(row, width, cre, ciRow, spacing, maxIter)
		return
	}
	var zr, zi, cr, ci, r2, n, lim [lanes]float64
	var px [lanes]int
	fd := feed{width: width}
	reset := func(k int) {
		zr[k], zi[k], n[k] = 0, 0, 0
		if x := fd.take(); x != idle {
			dcr := (float64(x) + 0.5 - 0.5*float64(width)) * spacing
			cr[k], ci[k], lim[k], px[k] = cre+dcr, ciRow, float64(maxIter), x
		} else { // idle lane: c = 0 never escapes
			cr[k], ci[k], lim[k], px[k] = 0, 0, math.Inf(1), idle
		}
	}
	for k := range lanes {
		reset(k)
	}
	two, one, esc := archsimd.BroadcastFloat64x4(2), archsimd.BroadcastFloat64x4(1), archsimd.BroadcastFloat64x4(escapeR2)
	for fd.active > 0 {
		// Iterate in registers until some lane finishes.
		zrL, zrH := load(&zr)
		ziL, ziH := load(&zi)
		crL, crH := load(&cr)
		ciL, ciH := load(&ci)
		nL, nH := load(&n)
		limL, limH := load(&lim)
		var r2L, r2H f64x4
		for {
			zrL, ziL, r2L = stepF64(zrL, ziL, crL, ciL, two)
			zrH, ziH, r2H = stepF64(zrH, ziH, crH, ciH, two)
			nL, nH = nL.Add(one), nH.Add(one)
			doneL := r2L.Greater(esc).Or(nL.GreaterEqual(limL))
			doneH := r2H.Greater(esc).Or(nH.GreaterEqual(limH))
			if doneL.Or(doneH).ToBits() != 0 {
				break
			}
		}
		store(&zr, zrL, zrH)
		store(&zi, ziL, ziH)
		store(&n, nL, nH)
		store(&r2, r2L, r2H)
		for k := range lanes {
			if px[k] == idle || !(r2[k] > escapeR2 || n[k] >= lim[k]) {
				continue
			}
			finish(row, px[k], n[k], r2[k])
			fd.active--
			reset(k)
		}
	}
}

// orbitAt loads the orbit entries ref[m[0..3]+off] as (re, im) vectors. Pairs are loaded
// straight from the orbit and shuffled in registers: writing them to an array and loading
// that as one vector would stall on store forwarding.
func orbitAt(ref [][2]float64, m []int, off int) (re, im f64x4) {
	x := f64x4{}.SetLo(archsimd.LoadFloat64x2(&ref[m[0]+off])).SetHi(archsimd.LoadFloat64x2(&ref[m[2]+off]))
	y := f64x4{}.SetLo(archsimd.LoadFloat64x2(&ref[m[1]+off])).SetHi(archsimd.LoadFloat64x2(&ref[m[3]+off]))
	return x.SelectFromPairGrouped(0, 2, y), x.SelectFromPairGrouped(1, 3, y)
}

// perturbCore and rebase are one iteration of dz <- (2Z_m + dz) dz + dc for lanes
// m[0..3], with z = Z_m+1 + dz checked for escape and rebasing. They are two functions
// so that each is small enough for Go to inline (as one, it would be called, and the
// vectors passed through memory).
func perturbCore(ref [][2]float64, m []int, dzr, dzi, dcr, dci, two f64x4) (nr, ni, zr, zi, r2 f64x4) {
	zmr, zmi := orbitAt(ref, m, 0)
	znr, zni := orbitAt(ref, m, 1)
	tr := two.Mul(zmr).Add(dzr)
	ti := two.Mul(zmi).Add(dzi)
	nr = tr.Mul(dzr).Sub(ti.Mul(dzi)).Add(dcr)
	ni = tr.Mul(dzi).Add(ti.Mul(dzr)).Add(dci)
	zr, zi = znr.Add(nr), zni.Add(ni)
	return nr, ni, zr, zi, zr.Mul(zr).Add(zi.Mul(zi))
}

// rebase (Zhuoran) restarts the reference when z gets closer to 0 than dz, or at the
// end of the orbit: dz = rb ? z : dz, m = rb ? 0 : m+1.
func rebase(nr, ni, zr, zi, r2, mf, one, lastm1 f64x4) (f64x4, f64x4, f64x4, f64x4, archsimd.Mask64x4) {
	rb := r2.Less(nr.Mul(nr).Add(ni.Mul(ni))).Or(mf.Equal(lastm1))
	return zr.Merge(nr, rb), zi.Merge(ni, rb), r2, f64x4{}.Merge(mf.Add(one), rb), rb
}

// rowPerturb iterates dz <- (2Z + dz) dz + dc around the reference orbit, with rebasing.
func rowPerturb(row []byte, width int, ref [][2]float64, dci, spacing float64, maxIter int32) {
	if !hasAVX2 {
		rowPerturbScalar(row, width, ref, dci, spacing, maxIter)
		return
	}
	last := len(ref) - 1
	var dzr, dzi, dcr, r2, n, lim, mf, live [lanes]float64 // live: 1 for lanes holding a pixel
	var m, px [lanes]int
	fd := feed{width: width}
	reset := func(k int) {
		dzr[k], dzi[k], n[k], m[k], mf[k] = 0, 0, 0, 0, 0
		if x := fd.take(); x != idle {
			dcr[k], lim[k], px[k], live[k] = (float64(x)+0.5-0.5*float64(width))*spacing, float64(maxIter), x, 1
		} else { // idle lane: follows the reference orbit and is never checked
			dcr[k], lim[k], px[k], live[k] = 0, math.Inf(1), idle, 0
		}
	}
	for k := range lanes {
		reset(k)
	}
	two, one, esc := archsimd.BroadcastFloat64x4(2), archsimd.BroadcastFloat64x4(1), archsimd.BroadcastFloat64x4(escapeR2)
	vdci, zero := archsimd.BroadcastFloat64x4(dci), archsimd.BroadcastFloat64x4(0)
	lastm1 := archsimd.BroadcastFloat64x4(float64(last - 1)) // m+1 == last
	for fd.active > 0 {
		dzrL, dzrH := load(&dzr)
		dziL, dziH := load(&dzi)
		dcrL, dcrH := load(&dcr)
		nL, nH := load(&n)
		limL, limH := load(&lim)
		mfL, mfH := load(&mf)
		liveL, liveH := load(&live)
		var r2L, r2H f64x4
		for {
			var rbL, rbH archsimd.Mask64x4
			nr, ni, zr, zi, r2 := perturbCore(ref, m[0:4], dzrL, dziL, dcrL, vdci, two)
			dzrL, dziL, r2L, mfL, rbL = rebase(nr, ni, zr, zi, r2, mfL, one, lastm1)
			nr, ni, zr, zi, r2 = perturbCore(ref, m[4:8], dzrH, dziH, dcrH, vdci, two)
			dzrH, dziH, r2H, mfH, rbH = rebase(nr, ni, zr, zi, r2, mfH, one, lastm1)
			rebase := rbL.ToBits() | rbH.ToBits()<<4
			for k := range lanes {
				if rebase&(1<<k) != 0 {
					m[k] = 0
				} else {
					m[k]++
				}
			}
			nL, nH = nL.Add(one), nH.Add(one)
			doneL := r2L.Greater(esc).And(liveL.Greater(zero)).Or(nL.GreaterEqual(limL))
			doneH := r2H.Greater(esc).And(liveH.Greater(zero)).Or(nH.GreaterEqual(limH))
			if doneL.Or(doneH).ToBits() != 0 {
				break
			}
		}
		store(&dzr, dzrL, dzrH)
		store(&dzi, dziL, dziH)
		store(&n, nL, nH)
		store(&r2, r2L, r2H)
		store(&mf, mfL, mfH)
		for k := range lanes {
			if px[k] == idle || !(r2[k] > escapeR2 || n[k] >= lim[k]) {
				continue
			}
			finish(row, px[k], n[k], r2[k])
			fd.active--
			reset(k)
		}
	}
}
