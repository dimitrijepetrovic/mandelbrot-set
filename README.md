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
cpp/        C++20, CMake; CPU: std::jthread + atomic row counter, auto-vectorized lanes; GPU: CUDA runtime <<<>>>
rust/       Rust 2024, cargo; CPU: rayon, auto-vectorized lanes; GPU: cudarc (driver API) + PTX built by build.rs
go/         Go 1.26; CPU: goroutines + atomic row counter, simd/archsimd lanes; GPU: cgo shim over the CUDA driver API
scripts/    bench.sh (timing table), verify.sh (cross-language output check)
tools/      misiurewicz.py (computes the default zoom centre to 340 digits)
```

The arbitrary-precision reference orbit uses each language's usual library: GMP (`gmpxx`) in
C++, `astro-float` in Rust, `math/big` in Go.

## Requirements

- g++ ≥ 13, CMake ≥ 3.24, GMP with C++ bindings (`libgmp-dev`)
- Rust ≥ 1.85 (edition 2024)
- Go ≥ 1.26 (for the experimental `simd/archsimd` package)
- CUDA toolkit (`nvcc`) and an NVIDIA driver; developed with CUDA 12.4 on an RTX 3080 Ti
- `ffmpeg` on `PATH`

## Build

```sh
make                    # builds bin/mandelbrot-cpp, bin/mandelbrot-rust, bin/mandelbrot-go
make cpp|rust|go        # one implementation
make CUDA_ARCH=sm_86    # target a specific GPU (default: the local one)
make CUDA_FMAD=false    # disable GPU fused multiply-add: bit-identical CPU vs GPU output
make NATIVE=false       # portable CPU code (see below)
```

By default the CPU code is built for the local CPU: `-march=native` (C++),
`-C target-cpu=native` (Rust, in `rust/.cargo/config.toml`) and `GOEXPERIMENT=simd` (Go, which
then uses AVX2 when the CPU has it). `NATIVE=false` builds for baseline x86-64, and Go then
iterates one pixel at a time (`go/cpu_nosimd.go`). Both builds produce the same frames.

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
scripts/bench.sh                         # all 12 combinations, 720p, 30 s, no video
scripts/bench.sh --duration 20           # extra flags go to every run
VIDEO=1 scripts/bench.sh                 # also write the videos to out/
LANGS="cpp go" DEVICES=gpu PRECISIONS=f64 scripts/bench.sh
```

Deep mode is expensive, especially on the CPU. Near the default centre, escape times grow with
depth, so every pixel takes thousands of iterations in the deep frames.

### Results

A single full run of `scripts/bench.sh` (1280×720, 900 frames, `--no-video`, run with
`nice -19` on an otherwise idle machine) on an Intel i9-12900H (20 threads) with an RTX 3080 Ti
Laptop GPU, default (`NATIVE=true`) build (2026-10-06):

| lang | device | precision | render_s | render_fps |
|------|--------|-----------|---------:|-----------:|
| C++  | cpu    | f64       |   27.07  |  33.25 |
| Rust | cpu    | f64       |   31.04  |  28.99 |
| Go   | cpu    | f64       |   36.45  |  24.69 |
| C++  | gpu    | f64       |   19.15  |  46.99 |
| Rust | gpu    | f64       |   19.15  |  47.01 |
| Go   | gpu    | f64       |   19.16  |  46.98 |
| C++  | cpu    | deep      |  496.17  |   1.81 |
| Rust | cpu    | deep      |  471.31* |   1.91 |
| Go   | cpu    | deep      |  533.75* |   1.69 |
| C++  | gpu    | deep      |  175.48  |   5.13 |
| Rust | gpu    | deep      |  176.85  |   5.09 |
| Go   | gpu    | deep      |  176.84  |   5.09 |

\* Rust and Go CPU deep were each re-run alone after the fixes described under "Rust deep"
and "Go deep" below.

The reference orbit takes under 0.01 s in every language, so it is negligible. On the GPU the
three languages are within 1% of each other, because they run the same kernel. In f64 the CPUs
now reach 53–71% of the GPU's speed. Deep mode gains less on the CPU, because each lane loads
its reference orbit entry from a different index on every step.

**CPU history.** Render fps over the same benchmark:

| precision | lang | original | bounds checks + SLP fix | 8 SIMD lanes + native |
|-----------|------|---------:|------------------------:|----------------------:|
| f64       | C++  |  11.97   | 12.86                   | **33.25**             |
| f64       | Rust |   8.27   | 11.48                   | **28.99**             |
| f64       | Go   |  11.03   | 11.56                   | **24.69**             |
| deep      | C++  |   1.43   |  1.49                   | **1.81**              |
| deep      | Rust |   1.21   |  1.45                   | 1.54 → **1.91**       |
| deep      | Go   |   1.21   |  1.34                   | 1.46 → **1.69**       |

1. **Bounds checks + SLP fix:** the Rust and Go perturbation loops did four bounds checks per
   iteration and now do one. Rust's f64 loop had been slowed by LLVM's SLP vectorizer packing
   the real and imaginary parts of one pixel. pprof showed 99% of Go's time in that loop.
2. **8 SIMD lanes + native:** a profile (callgrind for C++, pprof for Go) and a
   microbenchmark showed the one-pixel loop was latency-bound, at about 9.5 cycles per
   iteration. Each worker now iterates 8 pixels at once in SIMD registers (see
   [CPU lanes](#algorithm)), with `-march=native` / `target-cpu=native` / `GOEXPERIMENT=simd`.

**Rust deep.** Callgrind showed Rust doing only 3% more instructions than C++ in deep mode,
but taking 34% longer single-threaded, so the cost was stalls, not extra work. The
disassembly showed two overheads in the per-lane orbit loads:
- They did 16 bounds checks per step.
- The end-of-orbit flag (`m + 1 == last`) was built as 8 scalar bools and then packed into a
  vector mask through a chain of byte inserts and shuffles.

Each lane now loads `Z_m` and `Z_m+1` as one 2-element window, with one bounds check, and the
end test is done inside the vectorized loop as one vector compare. That took Rust deep from
1.54 to 1.91 fps, ahead of C++.

**Go deep.** pprof put about 30% of the time in the per-lane orbit loads and about 18% in a
branchy scalar loop updating each lane's `m`:
- The loads were 16 bounds-checked 128-bit loads per step.
- Rebases are frequent and unpredictable, so that loop's branches often mispredicted.

Now each lane's `Z_m, Z_m+1` comes in as one 4-wide load from the flat orbit, with one bounds
check, and the four lanes are transposed in registers. `m` lives in an integer vector, updated
with a vector select and stored once per step for the next step's loads. That took Go deep
from 1.46 to 1.69 fps.

**Remaining gap.** Go needs explicit SIMD because its compiler doesn't auto-vectorize, and the
Go compiler still keeps more values in memory than GCC or LLVM do.

### Fairness notes

- The three CPU versions do the same floating-point operations in the same order, which is what
  makes their output bit-identical. Fused multiply-add (FMA) is off everywhere on the CPU, even
  though `-march=native` makes it available. C++ uses `-ffp-contract=off`. Rust never fuses on
  its own. Go would fuse scalar `x*y + z` with `GOAMD64=v3`, so Go is built with the default
  `GOAMD64=v1`; its SIMD package uses AVX2 regardless.
- The CPU loops are vectorized differently. GCC and LLVM vectorize plain loops over the 8 lanes.
  Go doesn't auto-vectorize, so its lanes are written with the experimental `simd/archsimd`
  package as two 4-wide AVX2 vectors.
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

**CPU lanes.** One pixel's iterations form a serial dependency chain: each step needs the
previous `z`, so a single pixel leaves the core mostly idle. Each CPU worker therefore iterates
8 pixels of a row at once, in SIMD registers. When a lane escapes or reaches `max_iter`, its
pixel is coloured and the lane is refilled with the row's next pixel. In deep mode each step
first loads every lane's `Z_m` and `Z_m+1` (lanes are at different orbit indices), then does the
arithmetic in vectors, then updates `m`. Every lane does exactly the scalar operations of the
one-pixel loop, so the output is unchanged.

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
