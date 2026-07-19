from __future__ import annotations

import inspect
import unittest

import numpy as np

from genome_firewall.calibration import PlattCalibrator, select_decision_thresholds
from genome_firewall.train import train_models


class TrainingTests(unittest.TestCase):
    def test_training_does_not_load_held_out_test_artifacts(self) -> None:
        source = inspect.getsource(train_models)
        self.assertNotIn('_load_matrix(feature_dir, "test")', source)
        self.assertNotIn('_load_sample_ids(feature_dir, "test")', source)

    def test_platt_calibrator_returns_monotonic_probabilities(self) -> None:
        raw_scores = np.asarray([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0])
        labels = np.asarray([0, 0, 0, 1, 1, 1])
        calibrator = PlattCalibrator().fit(raw_scores, labels)
        probabilities = calibrator.predict_proba(raw_scores)
        self.assertTrue(np.all((probabilities >= 0.0) & (probabilities <= 1.0)))
        self.assertTrue(np.all(np.diff(probabilities) > 0.0))

    def test_threshold_selection_maximizes_valid_calls(self) -> None:
        probabilities = np.asarray([0.02, 0.04, 0.07, 0.10, 0.20, 0.75, 0.85, 0.90, 0.95, 0.99])
        labels = np.asarray([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        result = select_decision_thresholds(
            probabilities,
            labels,
            target_precision=0.95,
            minimum_calls=5,
            default_work_threshold=0.20,
            default_fail_threshold=0.80,
        )
        self.assertEqual(result["work"]["source"], "calibration")
        self.assertEqual(result["work"]["calls"], 5)
        self.assertEqual(result["work"]["threshold"], 0.20)
        self.assertEqual(result["fail"]["source"], "calibration")
        self.assertEqual(result["fail"]["calls"], 5)
        self.assertEqual(result["fail"]["threshold"], 0.75)

    def test_threshold_selection_falls_back_when_minimum_calls_is_unreachable(self) -> None:
        result = select_decision_thresholds(
            np.asarray([0.1, 0.9]),
            np.asarray([0, 1]),
            target_precision=0.95,
            minimum_calls=5,
            default_work_threshold=0.20,
            default_fail_threshold=0.80,
        )
        self.assertEqual(result["work"]["source"], "configured_default")
        self.assertEqual(result["fail"]["source"], "configured_default")


if __name__ == "__main__":
    unittest.main()
