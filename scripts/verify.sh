#!/usr/bin/env bash
# Renders a short low-res clip with all 12 variants (lossless) and compares them.
#   f64:  every language must produce bit-identical frames (per device).
#   deep: languages use different bignum libraries, so frames are compared by
#         PSNR against C++ instead (> 40 dB is visually identical).
# Build with `make CUDA_FMAD=false` to also get bit-identical CPU vs GPU output.
set -euo pipefail
cd "$(dirname "$0")/.."
out=out/verify
mkdir -p "$out"
args=(--width 480 --height 270 --fps 5 --duration 2 --encoder ffv1)
status=0

psnr() { ffmpeg -i "$1" -i "$2" -lavfi psnr -f null - 2>&1 | grep -o 'average:[^ ]*' | cut -d: -f2; }

for precision in f64 deep; do
  for device in cpu gpu; do
    declare -A sums=()
    for lang in cpp rust go; do
      file="$out/${lang}_${device}_${precision}.mkv"
      "bin/mandelbrot-$lang" --device "$device" --precision "$precision" "${args[@]}" \
        --output "$file" >/dev/null 2>&1
      sums[$lang]=$(ffmpeg -loglevel error -i "$file" -f md5 - | cut -d= -f2)
    done
    if [[ ${sums[cpp]} == "${sums[rust]}" && ${sums[cpp]} == "${sums[go]}" ]]; then
      result="bit-identical"
    elif [[ $precision == f64 ]]; then
      result="MISMATCH (cpp ${sums[cpp]:0:8} rust ${sums[rust]:0:8} go ${sums[go]:0:8})"
      status=1
    else
      result="PSNR vs cpp: rust $(psnr "$out/cpp_${device}_$precision.mkv" "$out/rust_${device}_$precision.mkv") dB," \
      result+=" go $(psnr "$out/cpp_${device}_$precision.mkv" "$out/go_${device}_$precision.mkv") dB"
    fi
    printf '%-5s %-4s %s\n' "$precision" "$device" "$result"
    unset sums
  done
  printf '%-5s cpu vs gpu (cpp): PSNR %s dB\n' "$precision" \
    "$(psnr "$out/cpp_cpu_$precision.mkv" "$out/cpp_gpu_$precision.mkv")"
done
exit $status
