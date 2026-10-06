//go:build !goexperiment.simd

package main

func rowF64(row []byte, width int, cre, ciRow, spacing float64, maxIter int32) {
	rowF64Scalar(row, width, cre, ciRow, spacing, maxIter)
}

func rowPerturb(row []byte, width int, ref [][2]float64, dci, spacing float64, maxIter int32) {
	rowPerturbScalar(row, width, ref, dci, spacing, maxIter)
}
