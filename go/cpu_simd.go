//go:build goexperiment.simd

package main

import (
	"math"
	"simd/archsimd"
	"unsafe"
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

type i64x4 = archsimd.Int64x4

// orbitAt loads Z_m and Z_m+1 for lanes m[0..3] as (re, im) vectors. Each lane's two
// entries are adjacent in the flat orbit, so they come in as one 4-wide load (one bounds
// check), and the four lanes are transposed in registers.
func orbitAt(flat []float64, m *[lanes]int64, base int) (zmr, zmi, znr, zni f64x4) {
	lane := func(k int) f64x4 {
		i := 2 * int(m[base+k])
		return archsimd.LoadFloat64x4((*[4]float64)(flat[i : i+4])) // Z_m.re, Z_m.im, Z_m+1.re, Z_m+1.im
	}
	a, b, c, d := lane(0), lane(1), lane(2), lane(3)
	t0, t1 := a.Select128FromPair(0, 2, c), b.Select128FromPair(0, 2, d) // Z_m of a, c / b, d
	t2, t3 := a.Select128FromPair(1, 3, c), b.Select128FromPair(1, 3, d) // Z_m+1
	return t0.SelectFromPairGrouped(0, 2, t1), t0.SelectFromPairGrouped(1, 3, t1),
		t2.SelectFromPairGrouped(0, 2, t3), t2.SelectFromPairGrouped(1, 3, t3)
}

// perturbCore and rebase are one iteration of dz <- (2Z_m + dz) dz + dc for 4 lanes,
// with z = Z_m+1 + dz checked for escape and rebasing. They are two functions so that
// each is small enough for Go to inline (as one, it would be called, and the vectors
// passed through memory).
func perturbCore(zmr, zmi, znr, zni, dzr, dzi, dcr, dci, two f64x4) (nr, ni, zr, zi, r2 f64x4) {
	tr := two.Mul(zmr).Add(dzr)
	ti := two.Mul(zmi).Add(dzi)
	nr = tr.Mul(dzr).Sub(ti.Mul(dzi)).Add(dcr)
	ni = tr.Mul(dzi).Add(ti.Mul(dzr)).Add(dci)
	zr, zi = znr.Add(nr), zni.Add(ni)
	return nr, ni, zr, zi, zr.Mul(zr).Add(zi.Mul(zi))
}

// rebase (Zhuoran) restarts the reference when z gets closer to 0 than dz, or at the
// end of the orbit: dz = rb ? z : dz, m = rb ? 0 : m+1.
func rebase(nr, ni, zr, zi, r2 f64x4, m, one, lastm1 i64x4) (f64x4, f64x4, i64x4) {
	rb := r2.Less(nr.Mul(nr).Add(ni.Mul(ni))).Or(m.Equal(lastm1))
	return zr.Merge(nr, rb), zi.Merge(ni, rb), i64x4{}.Merge(m.Add(one), rb)
}

// rowPerturb iterates dz <- (2Z + dz) dz + dc around the reference orbit, with rebasing.
func rowPerturb(row []byte, width int, ref [][2]float64, dci, spacing float64, maxIter int32) {
	if !hasAVX2 {
		rowPerturbScalar(row, width, ref, dci, spacing, maxIter)
		return
	}
	flat := unsafe.Slice(&ref[0][0], 2*len(ref)) // the orbit as re, im, re, im, ...
	last := len(ref) - 1
	var dzr, dzi, dcr, r2, n, lim, live [lanes]float64 // live: 1 for lanes holding a pixel
	var m [lanes]int64
	var px [lanes]int
	fd := feed{width: width}
	reset := func(k int) {
		dzr[k], dzi[k], n[k], m[k] = 0, 0, 0, 0
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
	oneI, lastm1 := archsimd.BroadcastInt64x4(1), archsimd.BroadcastInt64x4(int64(last-1)) // m+1 == last
	mLo, mHi := (*[4]int64)(m[0:4]), (*[4]int64)(m[4:8])
	for fd.active > 0 {
		dzrL, dzrH := load(&dzr)
		dziL, dziH := load(&dzi)
		dcrL, dcrH := load(&dcr)
		nL, nH := load(&n)
		limL, limH := load(&lim)
		liveL, liveH := load(&live)
		mL, mH := archsimd.LoadInt64x4(mLo), archsimd.LoadInt64x4(mHi)
		var r2L, r2H f64x4
		for {
			zmr, zmi, znr, zni := orbitAt(flat, &m, 0)
			nr, ni, zr, zi, r2 := perturbCore(zmr, zmi, znr, zni, dzrL, dziL, dcrL, vdci, two)
			dzrL, dziL, mL = rebase(nr, ni, zr, zi, r2, mL, oneI, lastm1)
			r2L = r2
			zmr, zmi, znr, zni = orbitAt(flat, &m, 4)
			nr, ni, zr, zi, r2 = perturbCore(zmr, zmi, znr, zni, dzrH, dziH, dcrH, vdci, two)
			dzrH, dziH, mH = rebase(nr, ni, zr, zi, r2, mH, oneI, lastm1)
			r2H = r2
			mL.Store(mLo) // the next step's orbit loads index with these
			mH.Store(mHi)
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
