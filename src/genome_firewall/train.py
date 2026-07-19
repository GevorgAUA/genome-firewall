"""Train per-antibiotic logistic models without using the held-out test split."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import sklearn
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression

from genome_firewall.calibration import PlattCalibrator, select_decision_thresholds
from genome_firewall.config import LoadedConfig, load_config

LABELS = {"susceptible": 0, "resistant": 1}


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _atomic_joblib(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    joblib.dump(value, temporary)
    os.replace(temporary, path)


def _atomic_coefficients(
    path: Path,
    feature_names: list[str],
    coefficients: NDArray[np.float64],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    rows = sorted(
        zip(feature_names, coefficients, strict=True),
        key=lambda item: (-abs(float(item[1])), item[0]),
    )
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["feature_name", "coefficient", "odds_ratio", "direction"],
        )
        writer.writeheader()
        for feature_name, coefficient in rows:
            value = float(coefficient)
            writer.writerow(
                {
                    "feature_name": feature_name,
                    "coefficient": value,
                    "odds_ratio": float(np.exp(value)),
                    "direction": "resistant" if value > 0 else "susceptible",
                }
            )
    os.replace(temporary, path)


def _load_sample_ids(feature_dir: Path, split: str) -> list[str]:
    path = feature_dir / f"sample_ids_{split}.csv"
    with path.open("r", encoding="utf-8", newline="") as handle:
        sample_ids = [row["sample_id"] for row in csv.DictReader(handle)]
    if not sample_ids or len(sample_ids) != len(set(sample_ids)):
        raise ValueError(f"Sample IDs for {split} are empty or duplicated")
    return sample_ids


def _load_matrix(feature_dir: Path, split: str) -> NDArray[np.uint8]:
    path = feature_dir / f"X_{split}.npz"
    with np.load(path) as archive:
        matrix = archive["X"]
    if matrix.ndim != 2:
        raise ValueError(f"Feature matrix must be two-dimensional: {path}")
    return np.asarray(matrix, dtype=np.uint8)


def _load_labels(
    manifest_path: Path,
    sample_ids: list[str],
    antibiotic: str,
) -> NDArray[np.uint8]:
    requested = set(sample_ids)
    labels_by_id: dict[str, int] = {}
    column = f"{antibiotic}_phenotype"
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if column not in (reader.fieldnames or []):
            raise ValueError(f"Manifest has no label column: {column}")
        for row in reader:
            genome_id = row["genome_id"]
            if genome_id not in requested:
                continue
            phenotype = row[column].strip().lower()
            if phenotype not in LABELS:
                raise ValueError(f"Unsupported {antibiotic} phenotype for {genome_id}: {phenotype}")
            labels_by_id[genome_id] = LABELS[phenotype]
    missing = requested - labels_by_id.keys()
    if missing:
        raise ValueError(f"Manifest is missing labels for {len(missing)} requested genomes")
    return np.asarray([labels_by_id[sample_id] for sample_id in sample_ids], dtype=np.uint8)


def _class_counts(labels: NDArray[np.uint8]) -> dict[str, int]:
    return {
        "susceptible": int((labels == 0).sum()),
        "resistant": int((labels == 1).sum()),
    }


def _validate_saved_artifacts(
    antibiotic_dir: Path,
    x_calibration: NDArray[np.uint8],
    feature_count: int,
) -> dict[str, Any]:
    """Reload persisted artifacts and verify their inference contract."""

    model = joblib.load(antibiotic_dir / "model.joblib")
    calibrator = joblib.load(antibiotic_dir / "calibrator.joblib")
    if model.coef_.shape != (1, feature_count):
        raise ValueError(f"Saved model has an invalid coefficient shape: {antibiotic_dir}")
    probabilities = np.asarray(
        calibrator.predict_proba(model.decision_function(x_calibration)),
        dtype=float,
    )
    if probabilities.shape != (x_calibration.shape[0],):
        raise ValueError(f"Saved calibrator has an invalid output shape: {antibiotic_dir}")
    if not np.isfinite(probabilities).all() or not np.all(
        (probabilities >= 0.0) & (probabilities <= 1.0)
    ):
        raise ValueError(f"Saved calibrator produced invalid probabilities: {antibiotic_dir}")
    return {
        "reload_succeeded": True,
        "coefficient_shape": list(model.coef_.shape),
        "calibration_probability_min": float(probabilities.min()),
        "calibration_probability_max": float(probabilities.max()),
    }


def train_models(config: LoadedConfig) -> dict[str, Any]:
    values = config.values
    feature_dir = config.resolve_path(values["paths"]["feature_dir"])
    model_dir = config.resolve_path(values["paths"]["model_dir"])
    manifest_path = config.resolve_path(values["paths"]["cohort_manifest"])
    model_settings = values["model"]
    decision_settings = values["decision"]

    feature_names = json.loads((model_dir / "feature_names.json").read_text(encoding="utf-8"))
    if not isinstance(feature_names, list) or not all(
        isinstance(name, str) for name in feature_names
    ):
        raise ValueError("feature_names.json must contain a string list")
    feature_hash = hashlib.sha256(
        json.dumps(feature_names, separators=(",", ":")).encode()
    ).hexdigest()

    # Deliberately load only fitting and threshold-tuning inputs. Test artifacts remain untouched.
    train_ids = _load_sample_ids(feature_dir, "train")
    calibration_ids = _load_sample_ids(feature_dir, "calibration")
    x_train = _load_matrix(feature_dir, "train")
    x_calibration = _load_matrix(feature_dir, "calibration")
    if x_train.shape != (len(train_ids), len(feature_names)):
        raise ValueError("Training matrix dimensions do not match sample IDs and features")
    if x_calibration.shape != (len(calibration_ids), len(feature_names)):
        raise ValueError("Calibration matrix dimensions do not match sample IDs and features")
    if set(train_ids) & set(calibration_ids):
        raise ValueError("Training and calibration sample IDs overlap")

    trained_at = datetime.now(UTC).isoformat()
    summaries: dict[str, Any] = {}
    for antibiotic in values["scope"]["primary_antibiotics"]:
        y_train = _load_labels(manifest_path, train_ids, antibiotic)
        y_calibration = _load_labels(manifest_path, calibration_ids, antibiotic)
        train_counts = _class_counts(y_train)
        calibration_counts = _class_counts(y_calibration)
        if len(y_train) < int(model_settings["minimum_training_samples"]):
            raise ValueError(f"Too few training samples for {antibiotic}")
        minimum_per_class = int(model_settings["minimum_samples_per_class"])
        if min(train_counts.values()) < minimum_per_class:
            raise ValueError(f"Too few training samples per class for {antibiotic}")
        if min(calibration_counts.values()) < minimum_per_class:
            raise ValueError(f"Too few calibration samples per class for {antibiotic}")

        model_arguments: dict[str, Any] = {
            "C": float(model_settings["C"]),
            "class_weight": model_settings["class_weight"],
            "solver": str(model_settings["solver"]),
            "max_iter": int(model_settings["max_iter"]),
            "random_state": int(model_settings["random_state"]),
        }
        # L2 is sklearn's default; omitting it avoids the 1.9 penalty deprecation warning.
        if str(model_settings["penalty"]) != "l2":
            model_arguments["penalty"] = str(model_settings["penalty"])
        model = LogisticRegression(
            **model_arguments,
        )
        model.fit(x_train, y_train)
        calibrator = PlattCalibrator(
            max_iter=int(model_settings["max_iter"]),
            random_state=int(model_settings["random_state"]),
        ).fit(model.decision_function(x_calibration), y_calibration)
        calibrated_probabilities = calibrator.predict_proba(
            model.decision_function(x_calibration)
        )
        thresholds = select_decision_thresholds(
            calibrated_probabilities,
            y_calibration,
            target_precision=float(decision_settings["target_precision"]),
            minimum_calls=int(decision_settings["minimum_calls_for_threshold"]),
            default_work_threshold=float(decision_settings["default_work_threshold"]),
            default_fail_threshold=float(decision_settings["default_fail_threshold"]),
        )

        antibiotic_dir = model_dir / antibiotic
        _atomic_joblib(antibiotic_dir / "model.joblib", model)
        _atomic_joblib(antibiotic_dir / "calibrator.joblib", calibrator)
        _atomic_json(antibiotic_dir / "thresholds.json", thresholds)
        _atomic_coefficients(
            antibiotic_dir / "coefficients.csv",
            feature_names,
            np.asarray(model.coef_[0], dtype=float),
        )
        metadata = {
            "antibiotic": antibiotic,
            "display_name": values["antibiotics"][antibiotic]["display_name"],
            "trained_at_utc": trained_at,
            "model_type": "logistic_regression",
            "calibration_method": "platt_on_held_out_calibration_split",
            "positive_class": "resistant",
            "feature_count": len(feature_names),
            "feature_names_sha256": feature_hash,
            "training_samples": len(train_ids),
            "training_class_counts": train_counts,
            "calibration_samples": len(calibration_ids),
            "calibration_class_counts": calibration_counts,
            "model_intercept": float(model.intercept_[0]),
            "calibration_slope": float(calibrator.estimator.coef_[0, 0]),
            "calibration_intercept": float(calibrator.estimator.intercept_[0]),
            "sklearn_version": sklearn.__version__,
            "random_state": int(model_settings["random_state"]),
            "test_split_used": False,
        }
        _atomic_json(antibiotic_dir / "metadata.json", metadata)
        artifact_validation = _validate_saved_artifacts(
            antibiotic_dir,
            x_calibration,
            len(feature_names),
        )
        summaries[antibiotic] = {
            "artifacts": str(antibiotic_dir.relative_to(config.project_root)),
            "artifact_validation": artifact_validation,
            "training_class_counts": train_counts,
            "calibration_class_counts": calibration_counts,
            "thresholds": thresholds,
        }

    summary = {
        "trained_at_utc": trained_at,
        "antibiotics": summaries,
        "feature_count": len(feature_names),
        "feature_names_sha256": feature_hash,
        "test_split_used": False,
    }
    _atomic_json(model_dir / "training_summary.json", summary)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    print(json.dumps(train_models(load_config(args.config)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
