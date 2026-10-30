import copy
import math
import unittest

from run_gemm import aggregate, geometric_mean, group_summary, validate_runs


def fixture(scale=1.0):
    document = {
        "status": "complete", "correctness": {"passed": True},
        "hardware": {"gpu": "test"}, "software": {"build_type": "Release"},
        "contract": {"dtype": "float32", "strict_sgemm_math": "CUBLAS_PEDANTIC_MATH",
                     "strict_lt_compute": "CUBLAS_COMPUTE_32F_PEDANTIC"},
        "options": {"suite": "quick", "cache": "hot", "samples": 3}, "cases": [],
    }
    for index, ratio in enumerate((0.25, 0.5, 1.0, 2.0, 4.0)):
        implementations = {}
        for name, factor in (("custom", 1), ("tiled_16x16", 3),
                             ("cublas_sgemm", ratio), ("cublaslt_tuned", ratio * 2),
                             ("cublaslt_legacy", 10)):
            latency = scale * factor * 10
            implementations[name] = {
                "samples_us": [latency * 0.9, latency, latency * 1.1],
                "median_us": latency, "cv": 0.01,
                "correctness": {"max_tolerance_ratio": 0.1, "max_abs_error": 0.001},
            }
        document["cases"].append({
            "name": str(index), "group": "square", "cache": "hot",
            "m": 128, "n": 128, "k": 128, "slots": 1,
            "working_set_bytes": 1, "target_reached": True,
            "implementations": implementations,
        })
    return document


class ReportTests(unittest.TestCase):
    def test_geometric_mean_does_not_use_arithmetic_mean(self):
        self.assertAlmostEqual(geometric_mean([0.25, 0.5, 1, 2, 4]), 1)

    def test_complete_case_set_and_stronger_blas_are_used(self):
        rows = aggregate([fixture(), fixture(2), fixture(3)])
        summary = group_summary(rows)
        self.assertEqual(len(rows), 5)
        self.assertAlmostEqual(rows[0]["latency_us"]["custom"], 20)
        self.assertAlmostEqual(summary["geomean_speedup_blas"], 1)
        self.assertAlmostEqual(summary["geomean_speedup_tiled"], 3)
        self.assertEqual(summary["wins_blas"], 2)

    def test_missing_or_incorrect_results_are_rejected(self):
        for mutation in ("missing_case", "wrong_median", "failed", "fast_math"):
            document = copy.deepcopy(fixture())
            if mutation == "missing_case":
                document["cases"].pop()
            elif mutation == "wrong_median":
                document["cases"][0]["implementations"]["custom"]["median_us"] = 1
            elif mutation == "failed":
                document["status"] = "failed"
            else:
                document["contract"]["strict_lt_compute"] = "CUBLAS_COMPUTE_32F_FAST_TF32"
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_runs([document])

    def test_invalid_latency_cannot_enter_summary(self):
        for value in (0, -1, math.nan, math.inf):
            with self.subTest(value=value), self.assertRaises(ValueError):
                geometric_mean([1, value])


if __name__ == "__main__":
    unittest.main()
