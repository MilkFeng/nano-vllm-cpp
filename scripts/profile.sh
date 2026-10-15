#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
report="${TMPDIR:-/tmp}/nano-profile-$(date +%Y%m%d-%H%M%S)"

cd "$project_root"

nsys profile \
  --trace=cuda,cublas,nvtx,osrt \
  --sample=none \
  --cpuctxsw=none \
  -o "$report" \
  bash "$project_root/scripts/start.sh"

nsys stats \
  --report cuda_gpu_kern_sum,cuda_api_sum,cuda_gpu_mem_time_sum \
  "$report.nsys-rep" | tee "$report-summary.txt"

printf '\nReport: %s\nSummary: %s\n' "$report.nsys-rep" "$report-summary.txt"
