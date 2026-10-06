#!/usr/bin/env bash
# Runs every language x device x precision combination and prints a table.
# Any arguments are passed to every run (they override the defaults below).
#   scripts/bench.sh                          720p, 5 s, render only (no video)
#   scripts/bench.sh --duration 20            longer run
#   VIDEO=1 scripts/bench.sh                  also encode the videos into out/
#   LANGS="cpp go" DEVICES=gpu PRECISIONS=f64 scripts/bench.sh
set -euo pipefail
cd "$(dirname "$0")/.."
LANGS=${LANGS:-"cpp rust go"}
DEVICES=${DEVICES:-"cpu gpu"}
PRECISIONS=${PRECISIONS:-"f64 deep"}
mkdir -p out
log="out/bench-$(date +%Y%m%d-%H%M%S).txt"

for precision in $PRECISIONS; do
  for device in $DEVICES; do
    for lang in $LANGS; do
      video=(--no-video)
      [[ ${VIDEO:-0} == 1 ]] && video=(--output "out/mandelbrot_${lang}_${device}_${precision}.mp4")
      echo ">> $lang $device $precision" >&2
      "bin/mandelbrot-$lang" --device "$device" --precision "$precision" \
        --width 1280 --height 720 --duration 5 "${video[@]}" "$@" | tee -a "$log"
    done
  done
done

echo
awk '
  { for (i = 1; i <= NF; i++) { split($i, kv, "="); v[kv[1]] = kv[2] } }
  NR == 1 { printf "%-5s %-4s %-5s %10s %8s %9s %9s %9s %11s\n", "lang", "dev", "prec", "size", "frames", "ref_s", "render_s", "total_s", "render_fps" }
  { printf "%-5s %-4s %-5s %10s %8s %9s %9s %9s %11s\n", v["lang"], v["device"], v["precision"], v["size"], v["frames"], v["ref_s"], v["render_s"], v["total_s"], v["render_fps"] }
' "$log"
echo "(raw results: $log)"
