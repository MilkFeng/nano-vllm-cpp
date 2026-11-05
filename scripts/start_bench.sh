#!/usr/bin/env bash

QWEN3_P6B_MODEL="$HOME/Desktop/PROGRAMS/cuda/models/Qwen3-0.6B"
QWEN35_35B_FP8_MODEL="$HOME/Desktop/PROGRAMS/cuda/models/Qwen3.5-35B-A3B-FP8"

# MODEL="$QWEN3_P6B_MODEL"
MODEL="$QWEN35_35B_FP8_MODEL"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

TIMESTAMP="$(date +%Y%m%d%H%M)"
OUTPUT_FILE="profile/qwen35_${TIMESTAMP}.json"

build/nano-vllm-cpp-bench \
    --model "$MODEL" \
    --input-tokens 128 \
    --output-tokens 128 \
    --batch-sizes 1 \
    --warmup 1 \
    --repeat 3 \
    --num-blocks 32 \
    --block-size 16 \
    --max-batched-tokens 128 \
    --cpu-threads 7 --cpu-affinity \
    --gpu-expert-ids 0,2,4,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64,66 \
    --expert-adapt-every 4 \
    --output "$OUTPUT_FILE"

exit $?
