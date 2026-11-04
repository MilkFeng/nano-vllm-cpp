"""Audit the complete FlashInfer matrix, raw samples, hashes, and 50% gate."""

import csv
import gzip
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics

DIRECTORY = Path(__file__).resolve().parent
ROOT = DIRECTORY.parents[3]


def read(name):
    path = DIRECTORY / name
    if path.exists():
        return json.loads(path.read_text())
    with gzip.open(str(path) + ".gz", "rt") as source:
        return json.load(source)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    spec = json.loads((ROOT / "exp/attn/suite.json").read_text())
    names = set()
    for profile in spec["profiles"]:
        for grid in spec["full"]["grids"]:
            lengths = grid.get("kv_lengths", grid.get("prefix_lengths", [0]))
            for batch, query, length in itertools.product(
                grid["batch_sizes"], grid["query_lengths"], lengths
            ):
                key = length if "kv_lengths" in grid else length + query
                names.add(f"{profile}_{grid['name']}_b{batch}_q{query}_k{key}")
        names.update(f"{profile}_{r['name']}" for r in spec["full"]["ragged"])
    expected = set(itertools.product(names, ("float16", "bfloat16"), ("reuse", "rotating")))
    assert len(expected) == 224
    manifest = read("manifest.json")
    assert len(manifest["processes"]) == 3
    assert all(p["exit_code"] == 0 for p in manifest["processes"])
    assert not manifest["source_changed_during_run"]
    assert manifest["source_sha256"] == manifest["source_sha256_after"]
    assert manifest["library_sha256"] == manifest["library_sha256_after"]
    for path, expected_hash in manifest["source_sha256"].items():
        assert digest(ROOT / path) == expected_hash, path
    assert digest(ROOT / "build/libnano-vllm-cpp-attn-exp.so") == manifest["library_sha256"]
    runs = []
    for index in range(1, 4):
        doc = read(f"run-{index:03}.json")
        assert doc["status"] == "complete"
        assert doc["expected_rows"] == 224 and len(doc["rows"]) == 224
        assert doc["source_sha256"] == doc["source_sha256_after"] == manifest["source_sha256"]
        assert doc["library_sha256"] == manifest["library_sha256"]
        options = doc["options"]
        assert options["samples"] == 15 and options["warmup"] == 20
        assert options["target_ms"] == 10 and options["seed"] == 41 + index
        assert options["suite"] == "full" and options["tasks"] == "pipeline"
        keyed = {}
        for row in doc["rows"]:
            key = row["case"]["name"], row["dtype"], row["cache"]
            assert key not in keyed and row["task"] == "pipeline"
            assert set(row["implementations"]) == {"custom", "flashinfer"}
            for result in row["implementations"].values():
                assert result["status"] == "ok"
                samples = result["samples_us"]
                assert len(samples) == 15 and all(math.isfinite(s) and s > 0 for s in samples)
                assert math.isclose(statistics.median(samples), result["median_us"], rel_tol=1e-12)
                checks = result["correctness"]
                assert checks["max_tolerance_ratio"] <= 1
                assert checks["fp64"]["max_tolerance_ratio"] <= 1
                assert result["metadata"]["append_included"]
            keyed[key] = row
        assert set(keyed) == expected
        runs.append(keyed)
    with (DIRECTORY / "summary.csv").open() as source:
        summary = list(csv.DictReader(source))
    assert len(summary) == 224
    groups = {}
    below = []
    for row in summary:
        key = row["name"], row["dtype"], row["cache"]
        assert row["status"] == "ok" and row["baseline"] == "flashinfer"
        timings = [statistics.median(r[key]["implementations"][name]["median_us"] for r in runs)
                   for name in ("custom", "flashinfer")]
        ratio = timings[1] / timings[0]
        assert math.isclose(float(row["custom_us"]), timings[0], rel_tol=1e-12)
        assert math.isclose(float(row["baseline_us"]), timings[1], rel_tol=1e-12)
        assert math.isclose(float(row["speedup"]), ratio, rel_tol=1e-12)
        group = tuple(row[k] for k in ("profile", "phase", "dtype", "cache"))
        groups.setdefault(group, []).append(ratio)
        if ratio < 0.5:
            below.append({"name": key[0], "dtype": key[1], "cache": key[2], "ratio": ratio})
    grouped = [{"profile": key[0], "phase": key[1], "dtype": key[2], "cache": key[3],
                "shapes": len(values), "ratio": math.exp(statistics.mean(map(math.log, values)))}
               for key, values in sorted(groups.items())]
    assert len(grouped) == 40
    assert read("correctness.json")["status"] == "ok"
    assert read("correctness.json")["checks"] == 212
    assert "Ran 15 tests" in (DIRECTORY / "unit-tests.txt").read_text()
    assert "\nOK\n" in (DIRECTORY / "unit-tests.txt").read_text()
    assert "PASS: 168" in (DIRECTORY / "memcheck.txt").read_text()
    assert "ERROR SUMMARY: 0 errors" in (DIRECTORY / "memcheck.txt").read_text()
    assert "PASS: 168" in (DIRECTORY / "racecheck.txt").read_text()
    assert "0 hazards displayed (0 errors, 0 warnings)" in (DIRECTORY / "racecheck.txt").read_text()
    failures = [g for g in grouped if g["ratio"] < 0.5]
    result = {"status": "passed" if not failures else "below_target",
              "metric": "FlashInfer median latency / custom median latency; equal-weight geometric mean within each group",
              "threshold": 0.5, "processes": 3, "cases_per_process": 224,
              "total_cases": 672, "coverage": "all 56 suite shapes x FP16/BF16 x reuse/rotating",
              "source_and_library_integrity": "passed", "correctness_and_sanitizers": "passed",
              "groups": grouped, "groups_below_target": failures,
              "individual_cases_below_target": below}
    (DIRECTORY / "target-check.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("groups", "individual_cases_below_target")}, indent=2))
    print(f"Individual cases below 50%: {len(below)}/224 (retained in target-check.json)")
    assert not failures, "some complete phase/dtype/cache groups remain below 50%"


if __name__ == "__main__":
    main()
