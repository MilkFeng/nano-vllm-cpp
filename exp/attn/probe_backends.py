#!/usr/bin/env python3
"""Check installed SDPA backends on small cases. This does not measure speed."""

import argparse
import datetime as dt
import importlib.metadata
import json
from pathlib import Path
import sys
import warnings


def package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import torch
    import torch.nn.functional as functional
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from torch.nn.attention.bias import causal_lower_right

    document = {
        "schema_version": 1,
        "purpose": "capability checks only; not performance results",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "python_executable": sys.executable,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "optional_packages": {},
        "trials": [],
    }

    for distribution in ("flash-attn", "flashinfer-python"):
        version = package_version(distribution)
        document["optional_packages"][distribution] = {
            "version": version,
            "status": "not_installed" if version is None else "not_probed",
            "note": "Paged-cache capability requires a separate package-specific probe.",
        }

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable in this Python environment")

    document["gpu"] = torch.cuda.get_device_name(0)
    document["capability"] = list(torch.cuda.get_device_capability(0))
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    expected_ops = {
        "math": "_scaled_dot_product_attention_math",
        "flash": "_scaled_dot_product_flash_attention",
        "cudnn": "_scaled_dot_product_cudnn_attention",
    }
    backends = {
        "math": SDPBackend.MATH,
        "flash": SDPBackend.FLASH_ATTENTION,
        "cudnn": SDPBackend.CUDNN_ATTENTION,
    }
    shapes = (("prefill", 128, 128), ("decode", 1, 512), ("chunked", 3, 20))

    with torch.inference_mode():
        for profile, hq, hkv, dimension in (
            ("qwen3", 16, 8, 128),
            ("qwen35_full", 16, 2, 256),
        ):
            for dtype in (torch.float32, torch.float16, torch.bfloat16):
                for phase, query_length, kv_length in shapes:
                    q = (
                        torch.randn(1, hq, query_length, dimension, device="cuda") * 0.5
                    ).to(dtype)
                    k = (
                        torch.randn(1, hkv, kv_length, dimension, device="cuda") * 0.5
                    ).to(dtype)
                    v = torch.randn_like(k)
                    mask = (
                        torch.arange(kv_length, device="cuda")[None, :]
                        <= kv_length
                        - query_length
                        + torch.arange(query_length, device="cuda")[:, None]
                    )

                    with sdpa_kernel([SDPBackend.MATH]):
                        reference = functional.scaled_dot_product_attention(
                            q.double(),
                            k.double(),
                            v.double(),
                            attn_mask=mask,
                            is_causal=False,
                            dropout_p=0.0,
                            enable_gqa=True,
                        )

                    for name, backend in backends.items():
                        trial = {
                            "profile": profile,
                            "dtype": str(dtype),
                            "phase": phase,
                            "query_length": query_length,
                            "kv_length": kv_length,
                            "requested_backend": name,
                        }
                        try:
                            with warnings.catch_warnings(record=True) as caught:
                                warnings.simplefilter("always")
                                with (
                                    sdpa_kernel([backend]),
                                    torch.profiler.profile(
                                        activities=[torch.profiler.ProfilerActivity.CPU]
                                    ) as trace,
                                ):
                                    if phase == "prefill":
                                        output = (
                                            functional.scaled_dot_product_attention(
                                                q,
                                                k,
                                                v,
                                                is_causal=True,
                                                dropout_p=0.0,
                                                enable_gqa=True,
                                            )
                                        )
                                    elif phase == "decode":
                                        output = (
                                            functional.scaled_dot_product_attention(
                                                q,
                                                k,
                                                v,
                                                is_causal=False,
                                                dropout_p=0.0,
                                                enable_gqa=True,
                                            )
                                        )
                                    else:
                                        output = (
                                            functional.scaled_dot_product_attention(
                                                q,
                                                k,
                                                v,
                                                attn_mask=causal_lower_right(
                                                    query_length, kv_length
                                                ),
                                                is_causal=False,
                                                dropout_p=0.0,
                                                enable_gqa=True,
                                            )
                                        )
                                    torch.cuda.synchronize()
                            ops = sorted(
                                {
                                    event.key
                                    for event in trace.key_averages()
                                    if "scaled_dot_product" in event.key
                                }
                            )
                            tolerance = {
                                torch.float32: (2e-5, 2e-4),
                                torch.float16: (2e-3, 1e-2),
                                torch.bfloat16: (1e-2, 3e-2),
                            }[dtype]
                            atol, rtol = tolerance
                            difference = (output.double() - reference).abs()
                            trial["observed_ops"] = ops
                            trial["warnings"] = [str(item.message) for item in caught]
                            trial["max_abs_error"] = difference.max().item()
                            trial["correct"] = bool(
                                torch.all(
                                    difference <= atol + rtol * reference.abs()
                                ).item()
                            )
                            verified = any(expected_ops[name] in op for op in ops)
                            trial["status"] = (
                                "ok"
                                if trial["correct"] and verified
                                else "incorrect_or_backend_mismatch"
                            )
                        except RuntimeError as error:
                            unsupported = (
                                "no available kernel",
                                "no viable backend",
                                "not supported",
                                "only supports",
                                "does not support",
                                "cannot support",
                                "not compiled with",
                                "no kernel image",
                            )
                            trial["status"] = (
                                "unsupported"
                                if any(
                                    reason in str(error).lower()
                                    for reason in unsupported
                                )
                                else "error"
                            )
                            trial["error"] = str(error)
                        document["trials"].append(trial)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    for name in backends:
        trials = [
            item for item in document["trials"] if item["requested_backend"] == name
        ]
        print(
            f"{name}: {sum(item['status'] == 'ok' for item in trials)}/{len(trials)} small cases available"
        )
    print(args.output)


if __name__ == "__main__":
    main()
