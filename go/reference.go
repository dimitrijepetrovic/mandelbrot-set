package main

import (
	"fmt"
	"math/big"
)

// referenceOrbit computes Z_0..Z_maxIter at the zoom centre with enough binary
// precision for the deepest frame, rounding each point to float64. It returns
// interleaved (re, im) pairs with Z_0 = 0.
func referenceOrbit(re, im string, maxIter int32, lnZoom float64) ([]float64, error) {
	prec := uint(lnZoom/ln2 + 64)
	cr, _, err := big.ParseFloat(re, 10, prec, big.ToNearestEven)
	if err != nil {
		return nil, fmt.Errorf("invalid centre: %w", err)
	}
	ci, _, err := big.ParseFloat(im, 10, prec, big.ToNearestEven)
	if err != nil {
		return nil, fmt.Errorf("invalid centre: %w", err)
	}

	newF := func() *big.Float { return new(big.Float).SetPrec(prec) }
	zr, zi, zr2, zi2, zrzi := newF(), newF(), newF(), newF(), newF()
	orbit := make([]float64, 2, 2*(int(maxIter)+1))
	for range maxIter {
		zr2.Mul(zr, zr)
		zi2.Mul(zi, zi)
		zrzi.Mul(zr, zi)
		zi.Add(zrzi, zrzi).Add(zi, ci)
		zr.Sub(zr2, zi2).Add(zr, cr)
		dr, _ := zr.Float64()
		di, _ := zi.Float64()
		orbit = append(orbit, dr, di)
		if dr*dr+di*di > escapeR2 {
			break // reference escaped: shorter orbit
		}
	}
	return orbit, nil
}
