import copy
import unittest

from cases import Case, expand_cases
from report import aggregate, geometric_mean
from run_attn import applicable


def fixture(scale=1.0):
    case = Case("test", "decode", "qwen3", 16, 8, 128, (1,), (128,)).document()

    def measurement(latency):
        return {
            "status": "ok",
            "samples_us": [latency * 0.9, latency, latency * 1.1],
            "median_us": latency,
            "cv": 0.01,
            "correctness": {"max_tolerance_ratio": 0.1, "max_abs_error": 1e-6},
        }

    return {
        "status": "complete",
        "expected_rows": 1,
        "hardware": {},
        "software": {},
        "contract": {},
        "options": {"samples": 3, "seed": 42, "workspace_mib": 256},
        "rows": [
            {
                "case": case,
                "task": "pipeline",
                "dtype": "float16",
                "cache": "reuse",
                "working_set": {"target_reached": True},
                "implementations": {
                    "custom": measurement(10 * scale),
                    "flashinfer": measurement(5 * scale),
                },
            }
        ],
    }


class AttentionTests(unittest.TestCase):
    def test_predeclared_case_counts(self):
        cases = expand_cases("full")
        self.assertEqual(len(cases), 56)
        self.assertEqual(len({case.name for case in cases}), 56)
        self.assertEqual(sum(applicable(case, "prefill") for case in cases), 12)
        self.assertEqual((56 + 12) * 3 * 2, 408)

    def test_invalid_shapes_are_rejected_before_gpu(self):
        for hq, hkv, q, k in ((7, 2, 1, 5), (8, 2, 6, 5), (8, 0, 1, 1)):
            with self.assertRaises(ValueError):
                Case("bad", "decode", "test", hq, hkv, 64, (q,), (k,))

    def test_aggregation_uses_matching_precision_and_contract(self):
        rows = aggregate([fixture(), fixture(2), fixture(3)])
        self.assertAlmostEqual(rows[0]["speedup"], 0.5)
        self.assertAlmostEqual(rows[0]["custom_us"], 20)
        self.assertAlmostEqual(geometric_mean([0.5, 2]), 1)
        changed = fixture()
        changed["rows"][0]["dtype"] = "float32"
        with self.assertRaises(ValueError):
            aggregate([fixture(), changed])

    def test_unsupported_does_not_turn_into_zero_or_fake_speedup(self):
        changed = fixture()
        changed["rows"][0]["implementations"]["flashinfer"] = {
            "status": "unsupported",
            "reason": "dtype",
        }
        row = aggregate([fixture(), changed])[0]
        self.assertEqual(row["status"], "unavailable")
        self.assertIsNone(row["speedup"])

    def test_bad_raw_samples_and_mixed_settings_are_rejected(self):
        for field in ("median", "workspace"):
            changed = copy.deepcopy(fixture())
            if field == "median":
                changed["rows"][0]["implementations"]["custom"]["median_us"] = 1
            else:
                changed["options"]["workspace_mib"] = 128
            with self.subTest(field=field), self.assertRaises(ValueError):
                aggregate([fixture(), changed])


if __name__ == "__main__":
    unittest.main()
