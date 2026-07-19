"""Evaluate frozen models once on the held-out test split."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    recall_score,
    roc_auc_score,
)

from genome_firewall.config import LoadedConfig, load_config
from genome_firewall.train import _load_labels, _load_matrix, _load_sample_ids


def wilson_interval(
    correct: int,
    total: int,
    z: float = 1.959963984540054,
) -> dict[str, float] | None:
    """Return a two-sided 95% Wilson score interval for a binomial proportion."""

    if total == 0:
        return None
    if not 0 <= correct <= total:
        raise ValueError("correct must be between zero and total")
    proportion = correct / total
    denominator = 1.0 + z**2 / total
    centre = (proportion + z**2 / (2.0 * total)) / denominator
    margin = (
        z
        * math.sqrt(proportion * (1.0 - proportion) / total + z**2 / (4.0 * total**2))
        / denominator
    )
    return {"lower": centre - margin, "upper": centre + margin}


def apply_decision_thresholds(
    probabilities: ArrayLike,
    work_threshold: float,
    fail_threshold: float,
) -> NDArray[np.str_]:
    """Map resistance probabilities to the frozen three-way model decision."""

    scores = np.asarray(probabilities, dtype=float)
    if not 0.0 <= work_threshold < fail_threshold <= 1.0:
        raise ValueError("Thresholds must satisfy 0 <= work < fail <= 1")
    decisions = np.full(scores.shape, "no_call", dtype="<U16")
    decisions[scores <= work_threshold] = "likely_to_work"
    decisions[scores >= fail_threshold] = "likely_to_fail"
    return decisions


def calculate_metrics(
    labels: ArrayLike,
    probabilities: ArrayLike,
    decisions: ArrayLike,
) -> dict[str, Any]:
    """Calculate required binary and abstaining-classifier metrics."""

    targets = np.asarray(labels, dtype=np.uint8)
    scores = np.asarray(probabilities, dtype=float)
    calls = np.asarray(decisions, dtype=str)
    if targets.shape != scores.shape or targets.shape != calls.shape or targets.ndim != 1:
        raise ValueError("Labels, probabilities, and decisions must be equal-length vectors")
    hard_predictions = (scores >= 0.5).astype(np.uint8)
    called = calls != "no_call"
    correct_called = ((calls == "likely_to_fail") & (targets == 1)) | (
        (calls == "likely_to_work") & (targets == 0)
    )
    work = calls == "likely_to_work"
    fail = calls == "likely_to_fail"
    correct_work = int(((targets == 0) & work).sum())
    correct_fail = int(((targets == 1) & fail).sum())
    called_count = int(called.sum())
    correct_called_count = int(correct_called.sum())

    return {
        "test_samples": int(targets.size),
        "resistant_samples": int((targets == 1).sum()),
        "susceptible_samples": int((targets == 0).sum()),
        "balanced_accuracy": float(balanced_accuracy_score(targets, hard_predictions)),
        "resistant_recall": float(recall_score(targets, hard_predictions, pos_label=1)),
        "susceptible_recall": float(recall_score(targets, hard_predictions, pos_label=0)),
        "f1": float(f1_score(targets, hard_predictions, pos_label=1)),
        "auroc": float(roc_auc_score(targets, scores)),
        "pr_auc": float(average_precision_score(targets, scores)),
        "brier_score": float(brier_score_loss(targets, scores)),
        "called_samples": called_count,
        "no_call_samples": int((~called).sum()),
        "no_call_rate": float((~called).mean()),
        "called_accuracy": correct_called_count / called_count if called_count else None,
        "called_accuracy_wilson_95": wilson_interval(correct_called_count, called_count),
        "likely_to_work_calls": int(work.sum()),
        "likely_to_work_precision": correct_work / int(work.sum()) if work.any() else None,
        "likely_to_work_precision_wilson_95": wilson_interval(correct_work, int(work.sum())),
        "likely_to_fail_calls": int(fail.sum()),
        "likely_to_fail_precision": correct_fail / int(fail.sum()) if fail.any() else None,
        "likely_to_fail_precision_wilson_95": wilson_interval(correct_fail, int(fail.sum())),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _atomic_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def flatten_metrics_for_csv(metrics: dict[str, Any]) -> dict[str, Any]:
    """Flatten interval objects while preserving the richer JSON representation."""

    flattened = dict(metrics)
    for field in (
        "called_accuracy_wilson_95",
        "likely_to_work_precision_wilson_95",
        "likely_to_fail_precision_wilson_95",
    ):
        interval = flattened.pop(field)
        flattened[f"{field}_lower"] = interval["lower"] if interval else None
        flattened[f"{field}_upper"] = interval["upper"] if interval else None
    return flattened


def _reliability_plot(
    path: Path,
    antibiotic: str,
    labels: NDArray[np.uint8],
    probabilities: NDArray[np.float64],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    observed, predicted = calibration_curve(labels, probabilities, n_bins=5, strategy="quantile")
    figure, axis = plt.subplots(figsize=(5.6, 4.4))
    axis.plot([0, 1], [0, 1], "--", color="#64748b", label="Perfect calibration")
    axis.plot(predicted, observed, "o-", color="#dc2626", linewidth=2, label=antibiotic)
    axis.set(
        xlim=(0, 1),
        ylim=(0, 1),
        xlabel="Mean predicted P(resistant)",
        ylabel="Observed resistant fraction",
    )
    axis.set_title(f"Reliability — {antibiotic.title()} (n={labels.size})")
    axis.grid(alpha=0.2)
    axis.legend(loc="best")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    figure.savefig(temporary, format="png", dpi=160)
    plt.close(figure)
    os.replace(temporary, path)


def evaluate_models(config: LoadedConfig) -> dict[str, Any]:
    values = config.values
    feature_dir = config.resolve_path(values["paths"]["feature_dir"])
    model_dir = config.resolve_path(values["paths"]["model_dir"])
    report_dir = config.resolve_path(values["paths"]["reports_dir"])
    manifest_path = config.resolve_path(values["paths"]["cohort_manifest"])

    test_ids = _load_sample_ids(feature_dir, "test")
    x_test = _load_matrix(feature_dir, "test")
    feature_names_path = model_dir / "feature_names.json"
    feature_names = json.loads(feature_names_path.read_text(encoding="utf-8"))
    if x_test.shape != (len(test_ids), len(feature_names)):
        raise ValueError("Test matrix dimensions do not match sample IDs and frozen features")

    evaluated_at = datetime.now(UTC).isoformat()
    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}

    for antibiotic in values["scope"]["primary_antibiotics"]:
        antibiotic_dir = model_dir / antibiotic
        model_path = antibiotic_dir / "model.joblib"
        calibrator_path = antibiotic_dir / "calibrator.joblib"
        thresholds_path = antibiotic_dir / "thresholds.json"
        metadata_path = antibiotic_dir / "metadata.json"
        artifact_hashes = {
            "model_joblib_sha256": _sha256(model_path),
            "calibrator_joblib_sha256": _sha256(calibrator_path),
            "thresholds_json_sha256": _sha256(thresholds_path),
            "metadata_json_sha256": _sha256(metadata_path),
            "feature_names_json_sha256": _sha256(feature_names_path),
        }
        model = joblib.load(model_path)
        calibrator = joblib.load(calibrator_path)
        thresholds = json.loads(thresholds_path.read_text(encoding="utf-8"))
        work_threshold = float(thresholds["work"]["threshold"])
        fail_threshold = float(thresholds["fail"]["threshold"])
        labels = _load_labels(manifest_path, test_ids, antibiotic)
        probabilities = np.asarray(
            calibrator.predict_proba(model.decision_function(x_test)),
            dtype=float,
        )
        decisions = apply_decision_thresholds(
            probabilities,
            work_threshold,
            fail_threshold,
        )
        metrics = calculate_metrics(labels, probabilities, decisions)
        metrics.update(
            {
                "antibiotic": antibiotic,
                "work_threshold": work_threshold,
                "fail_threshold": fail_threshold,
            }
        )
        metric_rows.append(flatten_metrics_for_csv(metrics))

        for sample_id, label, probability, decision in zip(
            test_ids,
            labels,
            probabilities,
            decisions,
            strict=True,
        ):
            is_called = decision != "no_call"
            is_correct = (decision == "likely_to_fail" and label == 1) or (
                decision == "likely_to_work" and label == 0
            )
            prediction_rows.append(
                {
                    "sample_id": sample_id,
                    "antibiotic": antibiotic,
                    "true_phenotype": "resistant" if label == 1 else "susceptible",
                    "p_resistant": float(probability),
                    "binary_prediction_at_0_5": (
                        "resistant" if probability >= 0.5 else "susceptible"
                    ),
                    "decision": decision,
                    "confidence": float(
                        probability if decision == "likely_to_fail" else 1.0 - probability
                    ) if is_called else float(max(probability, 1.0 - probability)),
                    "correct_if_called": bool(is_correct) if is_called else "",
                    "no_call_reason": "probability_in_uncertain_zone" if not is_called else "",
                    "work_threshold": work_threshold,
                    "fail_threshold": fail_threshold,
                }
            )

        plot_name = f"reliability_{antibiotic}.png"
        _reliability_plot(report_dir / plot_name, antibiotic, labels, probabilities)
        summaries[antibiotic] = {
            "metrics": metrics,
            "artifact_hashes_before_evaluation": artifact_hashes,
            "reliability_plot": plot_name,
        }

    metric_fields = [
        "antibiotic",
        "test_samples",
        "resistant_samples",
        "susceptible_samples",
        "balanced_accuracy",
        "resistant_recall",
        "susceptible_recall",
        "f1",
        "auroc",
        "pr_auc",
        "brier_score",
        "called_samples",
        "no_call_samples",
        "no_call_rate",
        "called_accuracy",
        "called_accuracy_wilson_95_lower",
        "called_accuracy_wilson_95_upper",
        "likely_to_work_calls",
        "likely_to_work_precision",
        "likely_to_work_precision_wilson_95_lower",
        "likely_to_work_precision_wilson_95_upper",
        "likely_to_fail_calls",
        "likely_to_fail_precision",
        "likely_to_fail_precision_wilson_95_lower",
        "likely_to_fail_precision_wilson_95_upper",
        "work_threshold",
        "fail_threshold",
    ]
    _atomic_csv(report_dir / "metrics.csv", metric_rows, metric_fields)
    _atomic_csv(
        report_dir / "predictions.csv",
        prediction_rows,
        list(prediction_rows[0]),
    )
    summary = {
        "evaluated_at_utc": evaluated_at,
        "evaluation_split": "test",
        "evaluation_scope": "frozen_probability_model_and_thresholds_before_target_gate",
        "models_or_thresholds_changed_during_evaluation": False,
        "test_genomes_per_antibiotic": len(test_ids),
        "antibiotics": summaries,
        "small_sample_warning": (
            "Empirical metrics are based on 24 test genomes per antibiotic; use the included "
            "Wilson intervals and do not interpret results as clinical validation."
        ),
    }
    _atomic_json(report_dir / "evaluation_summary.json", summary)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    print(json.dumps(evaluate_models(load_config(args.config)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
