from __future__ import annotations

import unittest

import numpy as np

from genome_firewall.evaluate import (
    apply_decision_thresholds,
    calculate_metrics,
    flatten_metrics_for_csv,
    wilson_interval,
)


class EvaluationTests(unittest.TestCase):
    def test_three_way_threshold_boundaries_are_inclusive(self) -> None:
        decisions = apply_decision_thresholds(
            np.asarray([0.1, 0.2, 0.21, 0.79, 0.8, 0.9]),
            work_threshold=0.2,
            fail_threshold=0.8,
        )
        self.assertEqual(
            decisions.tolist(),
            [
                "likely_to_work",
                "likely_to_work",
                "no_call",
                "no_call",
                "likely_to_fail",
                "likely_to_fail",
            ],
        )

    def test_required_metrics_respect_no_calls(self) -> None:
        labels = np.asarray([0, 0, 1, 1])
        probabilities = np.asarray([0.1, 0.4, 0.6, 0.9])
        decisions = np.asarray(["likely_to_work", "no_call", "no_call", "likely_to_fail"])
        metrics = calculate_metrics(labels, probabilities, decisions)
        self.assertEqual(metrics["balanced_accuracy"], 1.0)
        self.assertEqual(metrics["called_samples"], 2)
        self.assertEqual(metrics["no_call_rate"], 0.5)
        self.assertEqual(metrics["called_accuracy"], 1.0)

    def test_wilson_interval_is_honest_for_small_perfect_sample(self) -> None:
        interval = wilson_interval(5, 5)
        assert interval is not None
        self.assertLess(interval["lower"], 0.6)
        self.assertEqual(interval["upper"], 1.0)

    def test_metric_intervals_are_flattened_for_csv(self) -> None:
        metrics = calculate_metrics(
            np.asarray([0, 1]),
            np.asarray([0.1, 0.9]),
            np.asarray(["likely_to_work", "likely_to_fail"]),
        )
        flattened = flatten_metrics_for_csv(metrics)
        self.assertNotIn("called_accuracy_wilson_95", flattened)
        self.assertIn("called_accuracy_wilson_95_lower", flattened)
        self.assertIn("called_accuracy_wilson_95_upper", flattened)


if __name__ == "__main__":
    unittest.main()
