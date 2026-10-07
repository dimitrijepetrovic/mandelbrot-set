#!/usr/bin/env bash
# Renders a short low-res clip with all 12 variants, plus the GPU float32 deep mode
# (--gpu-fp32), losslessly and compares them across languages.
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

# check NAME PRECISION DEVICE [FLAGS...]: render with every language and compare.
check() {
  local name=$1 precision=$2 device=$3
  shift 3
  declare -A sums=()
  for lang in cpp rust go; do
    file="$out/${lang}_$name.mkv"
    "bin/mandelbrot-$lang" --device "$device" --precision "$precision" "${args[@]}" "$@" \
      --output "$file" >/dev/null 2>&1
    sums[$lang]=$(ffmpeg -loglevel error -i "$file" -f md5 - | cut -d= -f2)
  done
  if [[ ${sums[cpp]} == "${sums[rust]}" && ${sums[cpp]} == "${sums[go]}" ]]; then
    result="bit-identical"
  elif [[ $precision == f64 ]]; then
    result="MISMATCH (cpp ${sums[cpp]:0:8} rust ${sums[rust]:0:8} go ${sums[go]:0:8})"
    status=1
  else
    result="PSNR vs cpp: rust $(psnr "$out/cpp_$name.mkv" "$out/rust_$name.mkv") dB,"
    result+=" go $(psnr "$out/cpp_$name.mkv" "$out/go_$name.mkv") dB"
  fi
  printf '%-14s %s\n' "$name" "$result"
}

for precision in f64 deep; do
  for device in cpu gpu; do
    check "${precision}_$device" "$precision" "$device"
  done
  printf '%-14s cpu vs gpu (cpp): PSNR %s dB\n' "$precision" \
    "$(psnr "$out/cpp_${precision}_cpu.mkv" "$out/cpp_${precision}_gpu.mkv")"
done
check deep_gpu_fp32 deep gpu --gpu-fp32
printf '%-14s fp32 vs f64 gpu (cpp): PSNR %s dB\n' deep \
  "$(psnr "$out/cpp_deep_gpu.mkv" "$out/cpp_deep_gpu_fp32.mkv")"
exit $status
