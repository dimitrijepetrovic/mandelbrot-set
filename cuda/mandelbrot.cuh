// Shared Mandelbrot CUDA kernels. The same source is used by all three
// implementations: compiled directly into the C++ binary, and compiled to PTX
// for the Rust (cudarc) and Go (cgo + driver API) binaries.
#pragma once

extern "C" {

// Plain double-precision escape-time render. Writes width*height RGB triples.
__global__ void mandel_f64(unsigned char* rgb, int width, int height,
                           double centre_re, double centre_im, double spacing,
                           int max_iter);

// Perturbation render around a reference orbit `ref` (interleaved re,im pairs,
// ref_len entries, ref[0] == 0), with rebasing (Zhuoran) to avoid glitches.
__global__ void mandel_perturb(unsigned char* rgb, int width, int height,
                               const double* ref, int ref_len, double spacing,
                               int max_iter);

// The same perturbation render in float32, which consumer GPUs run up to 64x faster
// than float64. `ref` is the reference orbit rounded to float. Deltas start scaled by
// 2^-k (spacing = f * 2^k), so deep-zoom deltas below float's range still fit; valid
// while spacing >= MANDEL_F32_MIN_SPACING, below which callers use mandel_perturb.
// Output looks the same but is not bit-identical to the float64 kernels.
__global__ void mandel_perturb_f32(unsigned char* rgb, int width, int height,
                                   const float* ref, int ref_len, double spacing,
                                   int max_iter);
}

// 2^-220: the scaled deltas must reach 2^-100 (where they switch to plain float)
// without overflowing float, i.e. 2^(-100-k) <= 2^120.
#define MANDEL_F32_MIN_SPACING 5.9241243523004765e-67
