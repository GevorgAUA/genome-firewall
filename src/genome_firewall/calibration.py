"""Probability calibration and three-way decision threshold selection."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.linear_model import LogisticRegression


class PlattCalibrator:
    """Fit a sigmoid to a model's raw decision scores on held-out data."""

    def __init__(self, *, max_iter: int = 5000, random_state: int = 42) -> None:
        self.estimator = LogisticRegression(
            solver="liblinear",
            max_iter=max_iter,
            random_state=random_state,
        )

    def fit(self, raw_scores: ArrayLike, labels: ArrayLike) -> PlattCalibrator:
        scores = np.asarray(raw_scores, dtype=float).reshape(-1, 1)
        targets = np.asarray(labels, dtype=np.uint8)
        if scores.shape[0] != targets.shape[0]:
            raise ValueError("Calibration scores and labels must have equal lengths")
        if np.unique(targets).size != 2:
            raise ValueError("Platt calibration requires both phenotype classes")
        self.estimator.fit(scores, targets)
        return self

    def predict_proba(self, raw_scores: ArrayLike) -> NDArray[np.float64]:
        scores = np.asarray(raw_scores, dtype=float).reshape(-1, 1)
        return self.estimator.predict_proba(scores)[:, 1]


@dataclass(frozen=True)
class ThresholdResult:
    threshold: float
    source: str
    empirical_precision: float | None
    calls: int
    correct_calls: int


def _score_threshold(
    probabilities: NDArray[np.float64],
    labels: NDArray[np.uint8],
    threshold: float,
    *,
    resistant_call: bool,
) -> tuple[float | None, int, int]:
    called = probabilities >= threshold if resistant_call else probabilities <= threshold
    calls = int(called.sum())
    if calls == 0:
        return None, 0, 0
    expected = 1 if resistant_call else 0
    correct = int((labels[called] == expected).sum())
    return correct / calls, calls, correct


def _choose_threshold(
    probabilities: NDArray[np.float64],
    labels: NDArray[np.uint8],
    *,
    resistant_call: bool,
    target_precision: float,
    minimum_calls: int,
    default_threshold: float,
) -> ThresholdResult:
    candidates: list[ThresholdResult] = []
    for threshold in np.unique(probabilities):
        precision, calls, correct = _score_threshold(
            probabilities,
            labels,
            float(threshold),
            resistant_call=resistant_call,
        )
        if calls >= minimum_calls and precision is not None and precision >= target_precision:
            candidates.append(
                ThresholdResult(
                    threshold=float(threshold),
                    source="calibration",
                    empirical_precision=precision,
                    calls=calls,
                    correct_calls=correct,
                )
            )

    if candidates:
        # Coverage is primary; precision breaks ties. The final term makes ties deterministic.
        direction = -1.0 if resistant_call else 1.0
        return max(
            candidates,
            key=lambda result: (
                result.calls,
                result.empirical_precision or 0.0,
                direction * result.threshold,
            ),
        )

    precision, calls, correct = _score_threshold(
        probabilities,
        labels,
        default_threshold,
        resistant_call=resistant_call,
    )
    return ThresholdResult(
        threshold=default_threshold,
        source="configured_default",
        empirical_precision=precision,
        calls=calls,
        correct_calls=correct,
    )


def select_decision_thresholds(
    probabilities: ArrayLike,
    labels: ArrayLike,
    *,
    target_precision: float,
    minimum_calls: int,
    default_work_threshold: float,
    default_fail_threshold: float,
) -> dict[str, Any]:
    """Select high-precision WORK/FAIL cutoffs, leaving the middle uncertain."""

    scores = np.asarray(probabilities, dtype=float)
    targets = np.asarray(labels, dtype=np.uint8)
    if scores.ndim != 1 or scores.shape != targets.shape:
        raise ValueError("Probabilities and labels must be equal-length vectors")
    if scores.size == 0 or not np.isfinite(scores).all():
        raise ValueError("Probabilities must be a non-empty finite vector")
    if not 0.0 < target_precision <= 1.0:
        raise ValueError("target_precision must be in (0, 1]")
    if minimum_calls < 1:
        raise ValueError("minimum_calls must be at least one")
    if not 0.0 <= default_work_threshold < default_fail_threshold <= 1.0:
        raise ValueError("Default thresholds must satisfy 0 <= work < fail <= 1")

    work = _choose_threshold(
        scores,
        targets,
        resistant_call=False,
        target_precision=target_precision,
        minimum_calls=minimum_calls,
        default_threshold=default_work_threshold,
    )
    fail = _choose_threshold(
        scores,
        targets,
        resistant_call=True,
        target_precision=target_precision,
        minimum_calls=minimum_calls,
        default_threshold=default_fail_threshold,
    )

    fallback_reason: str | None = None
    if work.threshold >= fail.threshold:
        fallback_reason = "calibrated_thresholds_overlap"
        work_precision, work_calls, work_correct = _score_threshold(
            scores,
            targets,
            default_work_threshold,
            resistant_call=False,
        )
        fail_precision, fail_calls, fail_correct = _score_threshold(
            scores,
            targets,
            default_fail_threshold,
            resistant_call=True,
        )
        work = ThresholdResult(
            default_work_threshold,
            "configured_default",
            work_precision,
            work_calls,
            work_correct,
        )
        fail = ThresholdResult(
            default_fail_threshold,
            "configured_default",
            fail_precision,
            fail_calls,
            fail_correct,
        )

    return {
        "probability_definition": "probability_of_resistant_phenotype",
        "target_empirical_precision": target_precision,
        "minimum_calibration_calls": minimum_calls,
        "work": asdict(work),
        "uncertain": {
            "lower_exclusive": work.threshold,
            "upper_exclusive": fail.threshold,
        },
        "fail": asdict(fail),
        "fallback_reason": fallback_reason,
    }
