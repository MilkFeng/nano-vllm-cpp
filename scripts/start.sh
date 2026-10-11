#!/bin/bash

QWEN3_P6B_MODEL="$HOME/models/Qwen3-0.6B/"
QWEN35_35B_FP8_MODEL="$HOME/models/Qwen3.5-35B-A3B-FP8/"

MODEL=$QWEN35_35B_FP8_MODEL

build/nano-vllm-cpp \
  --model $MODEL \
  --prompt "你好" \
  --max-tokens 1024 \
  --num-blocks 256