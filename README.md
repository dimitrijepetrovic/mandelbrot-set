# Mandelbrot Set

A comparison of **C++**, **Rust** and **Go**, each rendering the same colour video zooming into the
Mandelbrot set. Each implementation can run on the **CPU** or on an NVIDIA **GPU** (CUDA), and in two
precision modes, all selected by command-line flags:

| Mode    | Arithmetic                                                     | Max zoom |
|---------|----------------------------------------------------------------|----------|
| `f64`   | plain double precision per pixel                               | ~1e12    |
| `deep`  | perturbation theory around an arbitrary-precision reference orbit | 1e300 |

All three implementations follow the same algorithm (see [Algorithm](#algorithm)) and produce
**bit-identical frames** (checked by `scripts/verify.sh`), so the timings compare the
languages and not different maths.

## Layout

```
cuda/       shared CUDA kernels (mandel_f64, mandel_perturb), used by all three
cpp/        C++20, CMake; CPU: std::jthread + atomic row counter; GPU: CUDA runtime <<<>>>
rust/       Rust 2024, cargo; CPU: rayon; GPU: cudarc (driver API) + PTX built by build.rs
go/         Go 1.26; CPU: goroutines + atomic row counter; GPU: cgo shim over the CUDA driver API
scripts/    bench.sh (timing table), verify.sh (cross-language output check)
tools/      misiurewicz.py (computes the default zoom centre to 340 digits)
```

The arbitrary-precision reference orbit uses each language's usual library: GMP (`gmpxx`) in
C++, `astro-float` in Rust, `math/big` in Go.

## Requirements

- g++ ≥ 13, CMake ≥ 3.24, GMP with C++ bindings (`libgmp-dev`)
- Rust ≥ 1.85 (edition 2024)
- Go ≥ 1.25
- CUDA toolkit (`nvcc`) and an NVIDIA driver; developed with CUDA 12.4 on an RTX 3080 Ti
- `ffmpeg` on `PATH`

## Build

```sh
make                    # builds bin/mandelbrot-cpp, bin/mandelbrot-rust, bin/mandelbrot-go
make cpp|rust|go        # one implementation
make CUDA_ARCH=sm_86    # target a specific GPU (default: the local one)
make CUDA_FMAD=false    # disable GPU fused multiply-add: bit-identical CPU vs GPU output
```

## Usage

All three binaries take the same flags:

```sh
bin/mandelbrot-rust --device gpu                       # 1080p30, 20 s, f64 zoom to 1e12
bin/mandelbrot-go   --device gpu --precision deep      # perturbation zoom to 1e50
bin/mandelbrot-cpp  --device cpu --precision deep --zoom 1e100 --duration 30
```

| Flag                | Default            | Meaning |
|---------------------|--------------------|---------|
| `--device`          | `cpu`              | `cpu` or `gpu` |
| `--precision`       | `f64`              | `f64` or `deep` |
| `--width --height`  | `1920 1080`        | frame size |
| `--fps`             | `30`               | frames per second |
| `--duration`        | `20`               | video length in seconds |
| `--zoom`            | `1e12` / `1e50`    | final magnification (f64 / deep default) |
| `--centre-re --centre-im` | Misiurewicz point | zoom centre as decimal strings (use many digits for deep zooms) |
| `--iter-base`       | `512`              | max iterations at zoom 1 |
| `--iter-per-decade` | `128`              | extra max iterations per 10× zoom |
| `--threads`         | all cores          | CPU worker threads |
| `--encoder`         | `libx264`          | any ffmpeg encoder, e.g. `h264_nvenc`, `libx265`, `ffv1` |
| `--output`          | `mandelbrot_<lang>_<device>_<precision>.mp4` | output file |
| `--no-video`        | off                | render only, skip ffmpeg (pure compute benchmark) |

Each run ends with one summary line on stdout:

```
lang=rust device=gpu precision=f64 size=1920x1080 frames=600 zoom=1e+12 ref_s=0.000 render_s=28.915 write_s=5.693 total_s=34.860 render_fps=20.75 output=...
```

`ref_s` is the reference orbit time (deep only), `render_s` is the frame computation time
(including the GPU→host copy), and `write_s` is time spent blocked on ffmpeg.

## Benchmarking

```sh
scripts/bench.sh                         # all 12 combinations, 720p, 5 s, no video
scripts/bench.sh --duration 20           # extra flags go to every run
VIDEO=1 scripts/bench.sh                 # also write the videos to out/
LANGS="cpp go" DEVICES=gpu PRECISIONS=f64 scripts/bench.sh
```

Deep mode is expensive, especially on the CPU. Near the default centre, escape times grow with
depth, so every pixel takes thousands of iterations in the deep frames.

### Results

A single run of `scripts/bench.sh` (1280×720, 150 frames, `--no-video`) on an Intel i9-12900H
(20 threads) with an RTX 3080 Ti Laptop GPU (2026-10-06):

| lang | device | precision | render_s | render_fps |
|------|--------|-----------|---------:|-----------:|
| C++  | cpu    | f64       |   26.07  |   5.75 |
| Rust | cpu    | f64       |   35.57  |   4.22 |
| Go   | cpu    | f64       |   25.67  |   5.84 |
| C++  | gpu    | f64       |    3.32  |  45.19 |
| Rust | gpu    | f64       |    3.27  |  45.85 |
| Go   | gpu    | f64       |    3.28  |  45.68 |
| C++  | cpu    | deep      |  189.77  |   0.79 |
| Rust | cpu    | deep      |  220.80  |   0.68 |
| Go   | cpu    | deep      |  223.32  |   0.67 |
| C++  | gpu    | deep      |   29.78  |   5.04 |
| Rust | gpu    | deep      |   30.02  |   5.00 |
| Go   | gpu    | deep      |   30.46  |   4.92 |

The reference orbit takes ≤ 0.013 s in every language, so it is negligible.

### Fairness notes

- All builds target the baseline x86-64 ISA (no `-march=native`, no FMA), so the three CPU
  versions do the same floating-point operations. This is also what makes their output
  bit-identical.
- All CPU versions split the work by rows with dynamic scheduling: an atomic counter in C++ and
  Go, rayon work stealing in Rust.
- The GPU kernel is the same PTX/SASS for all three, so GPU numbers mostly differ in host
  overhead (launch, copy, FFI). The GPU computes in FP64, which consumer GeForce cards run at
  1/64 of their FP32 rate.

## Algorithm

This is shared by every implementation. The constants live in `cuda/mandelbrot.cu`,
`cpp/src/common.hpp`, `rust/src/main.rs` and `go/main.go`, and must be kept in sync.

**Zoom schedule.** Frame `i` of `N` has `t = i/(N-1)`, magnification `zoom^t`, pixel spacing
`3 / (zoom^t · height)` (frame 0 is 3 units tall), and `max_iter = iter_base + iter_per_decade ·
t · log10(zoom)`. `exp`/`ln` here are small routines built only from correctly rounded IEEE
operations (`det_exp`/`det_ln`). The libm `pow`/`log` results differ in the last bit between Go
and C, and the chaotic iteration magnifies such a difference into visibly different pixels.

**f64 mode.** For each pixel `c = centre + offset`, iterate `z ← z² + c` from `z = 0` until
`|z|² > 65536` or `max_iter`.

**Deep mode.**
1. Compute the reference orbit `Z_n` at the centre once, with `log2(zoom) + 64` bits of
   precision, rounding each `Z_n` to nearest f64.
2. Each pixel iterates only its offset from the reference: `δz ← (2Z_m + δz)·δz + δc`, entirely
   in f64. The pixel offset `δc` is tiny (down to 1e-303) but well within f64 range.
3. Glitches are avoided by **rebasing** (Zhuoran): when `|Z_m + δz| < |δz|`, or the reference
   runs out, set `δz ← Z_m + δz` and restart at `m = 0`.

**Colour.** Smooth iteration count `ν = n + 1 − log2(ln|z|/ln 2)`, then a cosine palette
`0.5 + 0.5·cos(2π(0.015ν + φ))` with phases φ = 0, 0.15, 0.30 for R, G, B. Points that don't
escape are black.

**Default centre.** The Misiurewicz point M₂₃,₂ ≈ −0.77661 + 0.13461i (preperiod 23,
period 2), refined to 340 digits by `tools/misiurewicz.py`. Misiurewicz points sit on the
boundary of the set and have spiral detail at every scale, so the zoom never ends in a featureless
region, even at 1e300.

## Verification

```sh
scripts/verify.sh
```

`verify.sh` renders a short clip losslessly (`ffv1`) with all 12 variants and checks that C++,
Rust and Go are bit-identical for each device/precision pair. CPU vs GPU output differs slightly
by default (PSNR ≈ 30 dB for f64, ≈ 52 dB for deep), because nvcc fuses multiply-adds (FMA)
and the chaotic iteration amplifies the rounding differences. After `make CUDA_FMAD=false`,
all 12 variants produce identical frames, at the cost of about 30% GPU speed.
