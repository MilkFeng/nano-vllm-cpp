#!/usr/bin/env python3
"""Compare local C++ greedy generation with a local Transformers checkpoint.

Example:
  python tests/verify_generation.py --model ~/models/Qwen3-0.6B

This optional GPU integration check needs PyTorch and Transformers. It does not
download models, apply chat templates, or relax token equality for close logits.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time


DEFAULT_PROMPTS = (
    "The capital of France is",
    "The capital of France is Paris. The capital of Germany is",
    "1, 2, 3, 4,",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="local model directory")
    parser.add_argument(
        "--binary", type=Path,
        default=Path(__file__).resolve().parents[1] / "build" / "nano-vllm-cpp",
        help="C++ CLI executable (default: project build/nano-vllm-cpp)",
    )
    parser.add_argument("--prompt", action="append", help="raw prompt; may be repeated")
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--max-model-len", type=int, default=128)
    parser.add_argument("--timeout", type=float, default=300, help="seconds per C++ run")
    args = parser.parse_args()
    args.model = args.model.expanduser().resolve()
    args.binary = args.binary.expanduser().resolve()
    if not args.model.is_dir():
        parser.error(f"model directory does not exist: {args.model}")
    if not args.binary.is_file():
        parser.error(f"C++ executable does not exist: {args.binary}")
    if args.max_tokens < 1 or args.max_model_len < 2 or args.timeout <= 0:
        parser.error("max-tokens/timeout must be positive and max-model-len must be at least 2")
    args.prompts = args.prompt if args.prompt is not None else list(DEFAULT_PROMPTS)
    return args


def run_cpp(args: argparse.Namespace, name: str, batched_tokens: int,
            block_size: int, repeat: int) -> list[list[int]]:
    command = [
        str(args.binary), "--model", str(args.model),
        "--max-tokens", str(args.max_tokens), "--temperature", "0",
        "--max-model-len", str(args.max_model_len), "--max-seqs", "4",
        "--max-batched-tokens", str(batched_tokens), "--block-size", str(block_size),
        "--repeat", str(repeat), "--json",
    ]
    for prompt in args.prompts:
        command.extend(("--prompt", prompt))
    started = time.monotonic()
    print(f"Running C++ {name} ...", flush=True)
    try:
        result = subprocess.run(command, text=True, capture_output=True, timeout=args.timeout)
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(f"C++ {name} exceeded {args.timeout:g}s: {shlex.join(command)}") from error
    if result.returncode:
        raise RuntimeError(
            f"C++ {name} exited with status {result.returncode}\n"
            f"command: {shlex.join(command)}\n"
            f"stderr:\n{result.stderr[-12000:]}\nstdout:\n{result.stdout[-4000:]}"
        )
    try:
        outputs = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"C++ {name} did not return JSON: {result.stdout[-4000:]}") from error
    if not isinstance(outputs, list) or len(outputs) != len(args.prompts):
        raise RuntimeError(f"C++ {name} must return one JSON result per prompt")
    token_ids = []
    for index, output in enumerate(outputs):
        if not isinstance(output, dict) or not isinstance(output.get("text"), str):
            raise RuntimeError(f"C++ {name} result {index} has no text string")
        ids = output.get("token_ids")
        if (not isinstance(ids, list) or any(type(token) is not int or token < 0 for token in ids)
                or not 1 <= len(ids) <= args.max_tokens):
            raise RuntimeError(f"C++ {name} result {index} has invalid generated token_ids: {ids!r}")
        token_ids.append(ids)
    print(f"C++ {name} completed in {time.monotonic() - started:.2f}s", flush=True)
    return token_ids


def reference_generation(model, tokenizer, prompt_ids: list[int], max_tokens: int,
                         torch) -> tuple[list[int], list]:
    """Unpadded single-sequence greedy decoding, retaining logits for diagnostics."""
    generated = []
    logits_by_step = []
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device="cuda:0")
    past_key_values = None
    with torch.inference_mode():
        for step in range(max_tokens):
            attention_mask = torch.ones(
                (1, len(prompt_ids) + step), dtype=torch.long, device="cuda:0"
            )
            output = model(input_ids=input_ids, attention_mask=attention_mask,
                           past_key_values=past_key_values, use_cache=True, return_dict=True)
            logits = output.logits[0, -1].float()
            if not torch.isfinite(logits).all().item():
                raise RuntimeError("HF reference produced nonfinite logits")
            next_token = int(logits.argmax().item())
            logits_by_step.append(logits.cpu())
            generated.append(next_token)
            if next_token == tokenizer.eos_token_id:
                break
            past_key_values = output.past_key_values
            input_ids = torch.tensor([[next_token]], dtype=torch.long, device="cuda:0")
    return generated, logits_by_step


def first_difference(left: list[int], right: list[int]) -> int | None:
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return index
    return min(len(left), len(right)) if len(left) != len(right) else None


def describe_mismatch(name: str, prompt: str, actual: list[int], expected: list[int],
                      logits_by_step: list, tokenizer) -> str:
    step = first_difference(actual, expected)
    assert step is not None
    lines = [
        f"FAIL {name}, prompt={prompt!r}, first differing generated token={step}",
        f"  C++ IDs: {actual}",
        f"  HF  IDs: {expected}",
    ]
    if step >= len(logits_by_step):
        lines.append("  C++ continued after HF stopped at EOS; no further reference logits.")
        return "\n".join(lines)
    logits = logits_by_step[step]
    values, indices = logits.topk(2)
    first, second = int(indices[0]), int(indices[1])
    lines.append(
        f"  HF top two: {first} {tokenizer.decode([first])!r} ({float(values[0]):.7g}), "
        f"{second} {tokenizer.decode([second])!r} ({float(values[1]):.7g}); "
        f"margin={float(values[0] - values[1]):.7g}"
    )
    if step < len(actual):
        chosen = actual[step]
        if chosen < logits.numel():
            lines.append(
                f"  C++ choice under HF logits: {chosen} {tokenizer.decode([chosen])!r}, "
                f"logit={float(logits[chosen]):.7g}, "
                f"gap_to_HF_best={float(values[0] - logits[chosen]):.7g}"
            )
        else:
            lines.append(f"  C++ token {chosen} is outside the HF vocabulary.")
    else:
        lines.append("  C++ stopped before the reference.")
    lines.append("  Token equality is required even when the HF logit margin is small.")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("this integration check requires a CUDA GPU")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True,
                                             trust_remote_code=False)
    prompt_ids = [tokenizer.encode(prompt, add_special_tokens=False) for prompt in args.prompts]
    for prompt, ids in zip(args.prompts, prompt_ids):
        if not ids or len(ids) + args.max_tokens > args.max_model_len:
            raise RuntimeError(
                f"prompt must have 1..{args.max_model_len - args.max_tokens} tokens: {prompt!r}"
            )
    print(f"Model: {args.model}")
    print(f"Prompt lengths: {[len(ids) for ids in prompt_ids]}; max new tokens: {args.max_tokens}")
    print(f"PyTorch {torch.__version__}; Transformers {transformers.__version__}", flush=True)

    # Finish native processes before loading the HF model, so both runtimes can
    # use the GPU's available memory without competing for model/KV allocations.
    modes = {
        "ordinary prefill": (args.max_model_len, 4, 1),
        "chunked prefill": (3, 2, 1),
        "prefix reuse (repeat=2)": (3, 2, 2),
    }
    results = {name: run_cpp(args, name, *config) for name, config in modes.items()}
    failures = []
    ordinary = results["ordinary prefill"]
    for name, outputs in results.items():
        for index, (actual, baseline) in enumerate(zip(outputs, ordinary)):
            if actual != baseline:
                failures.append(
                    f"FAIL C++ modes disagree for prompt {index}: {name} {actual} "
                    f"vs ordinary prefill {baseline}"
                )

    print("Loading local HF reference (checkpoint dtype, eager attention) ...", flush=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, trust_remote_code=False,
        dtype="auto", attn_implementation="eager",
    ).to("cuda:0").eval()
    print(f"HF parameter dtype: {next(model.parameters()).dtype}", flush=True)
    for index, (prompt, ids) in enumerate(zip(args.prompts, prompt_ids)):
        expected, logits = reference_generation(model, tokenizer, ids, args.max_tokens, torch)
        for name, outputs in results.items():
            if outputs[index] != expected:
                failures.append(describe_mismatch(name, prompt, outputs[index], expected,
                                                   logits, tokenizer))
        print(f"HF prompt {index}: {expected} -> {tokenizer.decode(expected)!r}", flush=True)
    if failures:
        print("\n\n".join(failures), file=sys.stderr)
        return 1
    print(f"PASS: all {len(args.prompts)} prompts match HF greedy token IDs in all three C++ modes")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (RuntimeError, OSError, ImportError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
