#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Override defaults with environment variables, for example:
# EXPERT_ADAPT_EVERY=0 OUTPUT_FILE=profile/static.json bash scripts/start_bench.sh
MODEL="${MODEL:-"$HOME/models/Qwen3.5-35B-A3B-FP8"}"
BENCH_EXECUTABLE="${BENCH_EXECUTABLE:-"$PROJECT_ROOT/build/nano-vllm-cpp-bench"}"
PROMPTS_FILE="${PROMPTS_FILE:-"$PROJECT_ROOT/examples/benchmark-prompts.json"}"
INPUT_TOKENS="${INPUT_TOKENS:-512}"
OUTPUT_TOKENS="${OUTPUT_TOKENS:-512}"
BATCH_SIZES="${BATCH_SIZES:-1}"
WARMUP="${WARMUP:-2}"
REPEAT="${REPEAT:-5}"
NUM_BLOCKS="${NUM_BLOCKS:-128}"
BLOCK_SIZE="${BLOCK_SIZE:-16}"
MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-128}"
CPU_THREADS="${CPU_THREADS:-7}"
GPU_EXPERT_IDS="${GPU_EXPERT_IDS-0,2,4,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64,66,68,70,72,74,76,78,80,82,84,86,88,90,92,94,96,98,100}"
EXPERT_ADAPT_EVERY="${EXPERT_ADAPT_EVERY:-4}"
TIMESTAMP="$(date +%Y%m%d-%H%M%S)-$$"
OUTPUT_FILE="${OUTPUT_FILE:-"$PROJECT_ROOT/profile/benchmark_${TIMESTAMP}.json"}"
mkdir -p "$(dirname "$OUTPUT_FILE")"

bench_args=(
    --model "$MODEL"
    --input-tokens "$INPUT_TOKENS"
    --output-tokens "$OUTPUT_TOKENS"
    --batch-sizes "$BATCH_SIZES"
    --warmup "$WARMUP"
    --repeat "$REPEAT"
    --num-blocks "$NUM_BLOCKS"
    --block-size "$BLOCK_SIZE"
    --max-batched-tokens "$MAX_BATCHED_TOKENS"
    --cpu-threads "$CPU_THREADS"
    --expert-adapt-every "$EXPERT_ADAPT_EVERY"
    --seed 42
    --prompts-file "$PROMPTS_FILE"
    --output "$OUTPUT_FILE"
)
if [[ -n "$GPU_EXPERT_IDS" ]]; then
    bench_args+=(--gpu-expert-ids "$GPU_EXPERT_IDS")
fi

exec "$BENCH_EXECUTABLE" "${bench_args[@]}"
