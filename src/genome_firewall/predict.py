"""Run frozen Genome Firewall inference from FASTA or cached AMRFinder reports."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib

from genome_firewall.amrfinder_runner import (
    AnnotationItem,
    _run_one,
    annotation_paths,
    command_prefix,
    valid_amrfinder_tsv,
)
from genome_firewall.config import LoadedConfig, load_config
from genome_firewall.download_fastas import validate_fasta
from genome_firewall.evidence import collect_evidence, collect_model_contributions
from genome_firewall.features import encode_hit_sets
from genome_firewall.target_gate import assess_target

WARNING = (
    "Research prototype only. Confirm every prediction with validated laboratory "
    "antimicrobial susceptibility testing."
)


@lru_cache(maxsize=2)
def _load_feature_names(path: str) -> tuple[str, ...]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, list) or not all(isinstance(name, str) for name in value):
        raise ValueError("Frozen feature vocabulary is invalid")
    return tuple(value)


@lru_cache(maxsize=8)
def _load_model_bundle(directory: str) -> tuple[Any, Any, dict[str, Any]]:
    path = Path(directory)
    model = joblib.load(path / "model.joblib")
    calibrator = joblib.load(path / "calibrator.joblib")
    thresholds = json.loads((path / "thresholds.json").read_text(encoding="utf-8"))
    return model, calibrator, thresholds


def read_amrfinder_rows(path: Path) -> list[dict[str, str]]:
    """Read and minimally validate an AMRFinder report."""

    if not valid_amrfinder_tsv(path):
        raise ValueError(f"Invalid AMRFinder report: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _core_amr_symbols(rows: Iterable[Mapping[str, Any]]) -> set[str]:
    return {
        str(row.get("Element symbol", "")).strip()
        for row in rows
        if str(row.get("Scope", "")).strip().lower() == "core"
        and str(row.get("Type", "")).strip().upper() == "AMR"
        and str(row.get("Element symbol", "")).strip()
    }


def finalize_decision(
    probability: float,
    thresholds: Mapping[str, Any],
    target_status: str,
) -> tuple[str, str | None, float]:
    """Apply probability thresholds followed by the likely-to-work target gate."""

    work_threshold = float(thresholds["work"]["threshold"])
    fail_threshold = float(thresholds["fail"]["threshold"])
    if probability >= fail_threshold:
        return "likely_to_fail", None, probability
    if probability > work_threshold:
        return "no_call", "probability_in_uncertain_zone", max(probability, 1.0 - probability)
    if target_status == "absent":
        return "no_call", "target_absent", max(probability, 1.0 - probability)
    if target_status != "present":
        return "no_call", "target_unknown", max(probability, 1.0 - probability)
    return "likely_to_work", None, 1.0 - probability


def _failure_results(
    sample_id: str,
    config: LoadedConfig,
    antibiotics: list[str],
    reason: str,
) -> list[dict[str, Any]]:
    return [
        {
            "sample_id": sample_id,
            "species": config.values["scope"]["species_name"],
            "antibiotic": antibiotic,
            "display_name": config.values["antibiotics"]
            .get(antibiotic, {})
            .get("display_name", antibiotic),
            "prediction": "no_call",
            "p_resistant": None,
            "confidence": None,
            "target_status": "unknown",
            "target_evidence_source": None,
            "target_genes_found": [],
            "evidence_category": "no_known_resistance_signal",
            "evidence": [],
            "model_feature_contributions": [],
            "unseen_elements": [],
            "no_call_reason": reason,
            "warning": WARNING,
        }
        for antibiotic in antibiotics
    ]


def _selected_antibiotics(config: LoadedConfig, antibiotic: str | None) -> list[str]:
    supported = list(config.values["scope"]["primary_antibiotics"])
    if antibiotic is None:
        return supported
    normalized = antibiotic.strip().lower()
    if normalized not in supported:
        raise ValueError(f"Unsupported antibiotic: {antibiotic}")
    return [normalized]


def predict_from_reports(
    sample_id: str,
    amrfinder_report: Path,
    mutation_report: Path | None,
    config: LoadedConfig,
    antibiotic: str | None = None,
) -> list[dict[str, Any]]:
    """Predict from validated reports, enabling fast cached inference and testing."""

    antibiotics = _selected_antibiotics(config, antibiotic)
    main_rows = read_amrfinder_rows(amrfinder_report)
    mutation_rows = read_amrfinder_rows(mutation_report) if mutation_report else []
    detected_symbols = _core_amr_symbols(main_rows)
    model_dir = config.resolve_path(config.values["paths"]["model_dir"])
    feature_names = list(_load_feature_names(str(model_dir / "feature_names.json")))
    matrix, unseen = encode_hit_sets([detected_symbols], feature_names)
    unseen_symbols = sorted(unseen[0])

    results: list[dict[str, Any]] = []
    for selected in antibiotics:
        antibiotic_dir = model_dir / selected
        model, calibrator, thresholds = _load_model_bundle(str(antibiotic_dir))
        probability = float(
            calibrator.predict_proba(model.decision_function(matrix))[0]
        )
        settings = config.values["antibiotics"][selected]
        target = assess_target(
            settings.get("target_genes_any", []),
            main_rows,
            mutation_rows,
        )
        prediction, reason, confidence = finalize_decision(
            probability,
            thresholds,
            target["status"],
        )
        coefficients = dict(
            zip(feature_names, (float(value) for value in model.coef_[0]), strict=True)
        )
        known_evidence = collect_evidence(main_rows, settings, coefficients)
        if known_evidence:
            category = "known_resistance_signal"
        elif prediction == "likely_to_fail":
            category = "statistical_association_only"
        else:
            category = "no_known_resistance_signal"
        results.append(
            {
                "sample_id": sample_id,
                "species": config.values["scope"]["species_name"],
                "antibiotic": selected,
                "display_name": settings["display_name"],
                "prediction": prediction,
                "p_resistant": probability,
                "confidence": confidence,
                "target_status": target["status"],
                "target_evidence_source": target["source"],
                "target_genes_found": target["found_genes"],
                "evidence_category": category,
                "evidence": known_evidence,
                "model_feature_contributions": collect_model_contributions(
                    detected_symbols,
                    coefficients,
                ),
                "unseen_elements": unseen_symbols,
                "no_call_reason": reason,
                "warning": WARNING,
            }
        )
    return results


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_sample_id(fasta_path: Path, digest: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", fasta_path.stem).strip("._") or "uploaded"
    return f"{stem[:40]}_{digest[:12]}"


def predict_genome(
    fasta_path: str | Path,
    antibiotic: str | None = None,
    *,
    config_path: str | Path = "configs/mvp.yaml",
    timeout_seconds: int = 1200,
) -> list[dict[str, Any]]:
    """Validate a FASTA, run/cache AMRFinder, and return structured predictions."""

    config = load_config(config_path)
    path = Path(fasta_path).resolve()
    digest = _file_sha256(path) if path.is_file() else "missing"
    sample_id = _safe_sample_id(path, digest)
    try:
        antibiotics = _selected_antibiotics(config, antibiotic)
    except ValueError:
        requested = (antibiotic or "").strip().lower()
        return _failure_results(sample_id, config, [requested], "unsupported_antibiotic")
    try:
        validate_fasta(path)
    except (OSError, ValueError, UnicodeError):
        return _failure_results(sample_id, config, antibiotics, "invalid_fasta")

    output_root = config.resolve_path(config.values["paths"]["amrfinder_dir"])
    inference_dir = output_root / "inference" / digest[:16]
    item = AnnotationItem(sample_id, path)
    try:
        _run_one(
            config,
            command_prefix(config),
            item,
            force=False,
            timeout_seconds=timeout_seconds,
            output_dir_override=inference_dir,
        )
    except (OSError, RuntimeError, ValueError):
        return _failure_results(sample_id, config, antibiotics, "amrfinder_failed")
    reports = annotation_paths(inference_dir, sample_id)
    return predict_from_reports(
        sample_id,
        reports.output,
        reports.mutations,
        config,
        antibiotic,
    )


def predict_cohort_sample(
    sample_id: str,
    config: LoadedConfig,
    antibiotic: str | None = None,
) -> list[dict[str, Any]]:
    """Predict using an existing cohort sample's cached reports."""

    try:
        _selected_antibiotics(config, antibiotic)
    except ValueError:
        requested = (antibiotic or "").strip().lower()
        return _failure_results(sample_id, config, [requested], "unsupported_antibiotic")
    output_dir = config.resolve_path(config.values["paths"]["amrfinder_dir"])
    reports = annotation_paths(output_dir, sample_id)
    return predict_from_reports(sample_id, reports.output, reports.mutations, config, antibiotic)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/mvp.yaml")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--fasta")
    source.add_argument("--sample-id")
    parser.add_argument("--antibiotic")
    parser.add_argument("--timeout-seconds", type=int, default=1200)
    return parser


def main() -> None:
    args = _parser().parse_args()
    config = load_config(args.config)
    if args.fasta:
        results = predict_genome(
            args.fasta,
            args.antibiotic,
            config_path=args.config,
            timeout_seconds=args.timeout_seconds,
        )
    else:
        results = predict_cohort_sample(args.sample_id, config, args.antibiotic)
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
