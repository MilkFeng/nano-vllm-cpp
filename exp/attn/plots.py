"""Export diagnostic plots with standard plotting tools; no GPU work here."""

import os
from pathlib import Path


def export_plots(directory, rows):
    os.environ.setdefault(
        "MPLCONFIGDIR",
        str(Path(__file__).resolve().parents[2] / "build/attn-cache/matplotlib"),
    )
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plot
    except ImportError:
        return []
    exported = []
    profiles = ("qwen3_0p6b", "qwen35_full_attention")
    for filename, task, phase, baseline in (
        ("decode_latency", "pipeline", "decode", "flashinfer"),
        ("prefill_latency", "prefill", "prefill", "sdpa_flash"),
    ):
        chosen = [
            row
            for row in rows
            if row["task"] == task
            and row["phase"] == phase
            and row["dtype"] == "bfloat16"
            and row["baseline"] == baseline
            and row["status"] == "ok"
            and len(row["case"]["query_lengths"]) == 1
        ]
        if not chosen:
            continue
        figure, axes = plot.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
        for i, profile in enumerate(profiles):
            for j, cache in enumerate(("reuse", "rotating")):
                axis = axes[i][j]
                selected = sorted(
                    (
                        row
                        for row in chosen
                        if row["profile"] == profile and row["cache"] == cache
                    ),
                    key=lambda row: row["case"]["kv_lengths"][0],
                )
                if selected:
                    lengths = [row["case"]["kv_lengths"][0] for row in selected]
                    axis.plot(
                        lengths,
                        [row["custom_us"] for row in selected],
                        "o-",
                        label="Production attention",
                    )
                    axis.plot(
                        lengths,
                        [row["baseline_us"] for row in selected],
                        "s-",
                        label=baseline + " pipeline",
                    )
                axis.set_xscale("log", base=2)
                axis.set_yscale("log")
                axis.set_title(profile + " / " + cache)
                axis.set_xlabel("KV tokens after append")
                axis.set_ylabel("Latency (us, log scale)")
                axis.grid(True, which="both", alpha=0.2)
                axis.legend(fontsize=8)
        figure.suptitle("BF16, batch=1; append included; median across process medians")
        figure.savefig(directory / (filename + ".svg"))
        figure.savefig(directory / (filename + ".png"), dpi=150)
        plot.close(figure)
        exported.append(filename + ".svg")
    return exported
