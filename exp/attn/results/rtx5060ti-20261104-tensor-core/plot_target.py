"""Plot every verified phase/dtype/cache group against the 50% target."""

import json
import os
from pathlib import Path

DIRECTORY = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(DIRECTORY.parents[3] / "build/attn-cache/matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

groups = json.loads((DIRECTORY / "target-check.json").read_text())["groups"]
phases = ("decode", "prefill", "chunked_prefill", "ragged_decode", "ragged_chunked_prefill")
labels = ("Decode", "Prefill", "Chunked", "Ragged\ndecode", "Ragged\nchunked")
figure, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True, sharey=True)
for i, (profile, title) in enumerate((("qwen3_0p6b", "Qwen3: 16/8/128"),
                                     ("qwen35_full_attention", "Qwen3.5: 16/2/256"))):
    for j, dtype in enumerate(("float16", "bfloat16")):
        axis = axes[i, j]
        for cache, offset, color in (("reuse", -.19, "#2864a5"), ("rotating", .19, "#e18435")):
            values = [next(g["ratio"] * 100 for g in groups
                           if (g["profile"], g["dtype"], g["cache"], g["phase"])
                           == (profile, dtype, cache, phase)) for phase in phases]
            bars = axis.bar([k + offset for k in range(5)], values, .36, label=cache, color=color)
            axis.bar_label(bars, fmt="%.1f", fontsize=7, padding=3)
        axis.axhline(50, color="#992c2c", linestyle="--", linewidth=1.2, label="50% target")
        axis.axhline(100, color="gray", linestyle=":", linewidth=.8)
        axis.set_xticks(range(5), labels, fontsize=8)
        axis.set_title(title + " / " + ("FP16" if dtype == "float16" else "BF16"))
        axis.set_ylim(0, 125)
        axis.set_ylabel("Throughput / FlashInfer (%)")
        axis.grid(axis="y", alpha=.18)
        axis.set_axisbelow(True)
figure.legend(*axes[0, 0].get_legend_handles_labels(), loc="outside lower center", ncol=3, fontsize=9)
figure.suptitle("Paged KV append + attention: all 40 groups exceed 50%\n"
               "RTX 5060 Ti | 3 processes | equal-weight shape geometric means", fontsize=12)
figure.savefig(DIRECTORY / "throughput_target.svg")
figure.savefig(DIRECTORY / "throughput_target.png", dpi=160)
