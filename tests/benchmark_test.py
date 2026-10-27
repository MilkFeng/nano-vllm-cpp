#!/usr/bin/env python3
"""GPU integration checks for stage accounting; no Python model dependencies."""

import argparse
import json
import math
from pathlib import Path
import statistics
import subprocess
import tempfile


def close(actual, expected):
    assert math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9), (
        actual, expected
    )


def check_report(report, repeat, *, recompute=False):
    assert report["completed"] is True
    assert report["settings"]["prefix_cache"] is False
    for case in report["cases"]:
        batch = case["batch_size"]
        inputs = case["input_tokens"]
        outputs = case["output_tokens"]
        assert len(case["runs"]) == repeat  # Warmup must not appear in samples.
        ttfts, tpots = [], []
        for run in case["runs"]:
            prefill, decode = run["prefill"], run["decode"]
            assert run["input_tokens"] == batch * inputs
            assert run["output_tokens"] == batch * outputs
            assert prefill["output_tokens"] + decode["output_tokens"] == batch * outputs
            assert prefill["steps"] >= 1
            assert prefill["ms"] > 0
            assert prefill["ms"] + decode["ms"] <= run["inference_ms"]
            assert run["inference_ms"] <= run["total_ms"]
            close(prefill["input_tokens_per_second"],
                  prefill["input_tokens"] * 1000 / prefill["ms"])
            close(run["output_tokens_per_second"],
                  batch * outputs * 1000 / run["inference_ms"])
            if recompute:
                assert prefill["input_tokens"] > batch * inputs
                assert prefill["output_tokens"] > batch
            else:
                # Repeated identical prompts must still execute every input token.
                assert prefill["input_tokens"] == batch * inputs
                assert prefill["output_tokens"] == batch
                assert decode["output_tokens"] == batch * (outputs - 1)
                assert decode["steps"] == outputs - 1
            if outputs == 1:
                assert decode["ms"] == 0
                assert decode["output_tokens_per_second"] is None
                assert decode["ms_per_step"] is None
            else:
                close(decode["output_tokens_per_second"],
                      decode["output_tokens"] * 1000 / decode["ms"])
                close(decode["ms_per_step"], decode["ms"] / decode["steps"])
            assert len(run["requests"]) == batch
            for request in run["requests"]:
                assert request["input_tokens"] == inputs
                assert request["output_tokens"] == outputs
                assert 0 < request["ttft_ms"] <= request["latency_ms"]
                assert request["latency_ms"] <= run["inference_ms"]
                ttfts.append(request["ttft_ms"])
                if outputs == 1:
                    assert request["tpot_ms"] is None
                    close(request["ttft_ms"], request["latency_ms"])
                else:
                    close(request["tpot_ms"],
                          (request["latency_ms"] - request["ttft_ms"]) / (outputs - 1))
                    tpots.append(request["tpot_ms"])
        summary = case["summary"]
        for name, samples in (("ttft_ms", ttfts), ("tpot_ms", tpots)):
            values = summary[name]
            assert values["count"] == len(samples)
            if not samples:
                assert values["mean"] is None
                continue
            close(values["mean"], statistics.mean(samples))
            close(values["stddev"], statistics.pstdev(samples))
            for label, fraction in (("p50", .5), ("p95", .95)):
                ordered = sorted(samples)
                index = fraction * (len(ordered) - 1)
                lo, hi = math.floor(index), math.ceil(index)
                close(values[label], ordered[lo] + (ordered[hi] - ordered[lo]) * (index - lo))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", default="build/nano-vllm-cpp-bench")
    parser.add_argument("--model", required=True,
                        help="Small supported model, e.g. Qwen3-0.6B")
    args = parser.parse_args()
    executable = str(Path(args.executable).resolve())
    with tempfile.TemporaryDirectory(prefix="nanovllm-benchmark-") as directory:
        for name, options, cases, recompute in (
            ("matrix", ["--input-tokens", "17,33", "--output-tokens", "1,5",
                        "--batch-sizes", "1,2", "--num-blocks", "32",
                        "--max-batched-tokens", "16"], 8, False),
            ("preemption", ["--input-tokens", "17", "--output-tokens", "12",
                            "--batch-sizes", "2", "--num-blocks", "6",
                            "--max-batched-tokens", "8"], 1, True),
        ):
            output = Path(directory) / f"{name}.json"
            command = [executable, "--model", args.model, "--block-size", "8",
                       "--warmup", "1", "--repeat", "2", "--output", str(output),
                       *options]
            result = subprocess.run(command, capture_output=True, text=True, timeout=180)
            if result.returncode:
                raise RuntimeError(f"{name}: {result.stdout}\n{result.stderr}")
            report = json.loads(output.read_text(encoding="utf-8"))
            assert len(report["cases"]) == cases
            check_report(report, 2, recompute=recompute)
            print(f"{name}: passed ({cases} cases)")
    print("Benchmark accounting checks passed.")


if __name__ == "__main__":
    main()
