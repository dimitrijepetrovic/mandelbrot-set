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
}
