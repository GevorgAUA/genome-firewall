"""Build train-fitted binary AMRFinder feature matrices."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from genome_firewall.config import LoadedConfig, load_config


def extract_element_symbols(path: Path, scope: str = "core", row_type: str = "AMR") -> set[str]:
    """Extract unique biological element symbols from one AMRFinder main report."""

    symbols: set[str] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"Element symbol", "Scope", "Type"}
        if not required <= set(reader.fieldnames or []):
            raise ValueError(f"AMRFinder report has an invalid header: {path}")
        for row in reader:
            if (row.get("Scope") or "").strip().lower() != scope.lower():
                continue
            if (row.get("Type") or "").strip().upper() != row_type.upper():
                continue
            symbol = (row.get("Element symbol") or "").strip()
            if symbol:
                symbols.add(symbol)
    return symbols


def build_vocabulary(training_hits: Iterable[set[str]]) -> list[str]:
    """Fit the stable vocabulary using training hit sets only."""

    vocabulary: set[str] = set()
    for hits in training_hits:
        vocabulary.update(hits)
    return sorted(vocabulary)


def encode_hit_sets(
    hit_sets: list[set[str]],
    feature_names: list[str],
) -> tuple[np.ndarray, list[set[str]]]:
    """Encode sets into a fixed binary matrix and return unseen elements per sample."""

    index = {name: column for column, name in enumerate(feature_names)}
    matrix = np.zeros((len(hit_sets), len(feature_names)), dtype=np.uint8)
    unseen: list[set[str]] = []
    for row_number, hits in enumerate(hit_sets):
        unknown: set[str] = set()
        for symbol in hits:
            column = index.get(symbol)
            if column is None:
                unknown.add(symbol)
            else:
                matrix[row_number, column] = 1
        unseen.append(unknown)
    return matrix, unseen


def _read_manifest(config: LoadedConfig) -> list[dict[str, str]]:
    path = config.resolve_path(config.values["paths"]["cohort_manifest"])
    if not path.is_file():
        raise FileNotFoundError(f"Cohort manifest not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Cohort manifest is empty: {path}")
    genome_ids = [row["genome_id"] for row in rows]
    if len(genome_ids) != len(set(genome_ids)):
        raise ValueError("Cohort manifest contains duplicate genome IDs")
    return rows


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty sample ID file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id"])
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _atomic_npz(path: Path, matrix: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, X=matrix)
    os.replace(temporary, path)


def build_features(config: LoadedConfig) -> dict[str, Any]:
    values = config.values
    manifest = _read_manifest(config)
    feature_dir = config.resolve_path(values["paths"]["feature_dir"])
    model_dir = config.resolve_path(values["paths"]["model_dir"])
    amrfinder_dir = config.resolve_path(values["paths"]["amrfinder_dir"])
    feature_settings = values["features"]

    hit_sets: dict[str, set[str]] = {}
    for row in manifest:
        genome_id = row["genome_id"]
        report = amrfinder_dir / f"{genome_id}.tsv"
        if not report.is_file():
            raise FileNotFoundError(f"AMRFinder report not found: {report}")
        hit_sets[genome_id] = extract_element_symbols(
            report,
            scope=str(feature_settings["scope"]),
            row_type=str(feature_settings["type"]),
        )

    split_rows = {
        split: sorted(
            (row for row in manifest if row["split"] == split),
            key=lambda row: row["genome_id"],
        )
        for split in ("train", "calibration", "test")
    }
    if any(not rows for rows in split_rows.values()):
        raise ValueError("Train, calibration, and test splits must all be non-empty")

    feature_names = build_vocabulary(
        hit_sets[row["genome_id"]] for row in split_rows["train"]
    )
    if not feature_names:
        raise ValueError("Training genomes produced an empty AMRFinder feature vocabulary")

    _atomic_json(model_dir / "feature_names.json", feature_names)
    feature_hash = hashlib.sha256(
        json.dumps(feature_names, separators=(",", ":")).encode()
    ).hexdigest()

    split_summary: dict[str, Any] = {}
    unseen_summary: dict[str, Any] = {}
    for split, rows in split_rows.items():
        samples = [row["genome_id"] for row in rows]
        matrix, unseen = encode_hit_sets(
            [hit_sets[sample_id] for sample_id in samples],
            feature_names,
        )
        _atomic_npz(feature_dir / f"X_{split}.npz", matrix)
        _atomic_csv(
            feature_dir / f"sample_ids_{split}.csv",
            [{"sample_id": sample_id} for sample_id in samples],
        )
        unseen_counts = Counter(symbol for symbols in unseen for symbol in symbols)
        unseen_summary[split] = {
            "unique_elements": sorted(unseen_counts),
            "sample_counts": dict(sorted(unseen_counts.items())),
            "samples_with_unseen_elements": sum(bool(symbols) for symbols in unseen),
        }
        split_summary[split] = {
            "samples": matrix.shape[0],
            "features": matrix.shape[1],
            "nonzero_values": int(matrix.sum()),
            "density": float(matrix.mean()),
            "all_zero_samples": int(np.sum(matrix.sum(axis=1) == 0)),
        }

    summary = {
        "matrix_format": "dense_numpy_uint8_npz",
        "matrix_key": "X",
        "feature_count": len(feature_names),
        "feature_names_sha256": feature_hash,
        "vocabulary_source_split": "train",
        "splits": split_summary,
        "unseen": unseen_summary,
    }
    _atomic_json(feature_dir / "feature_summary.json", summary)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = build_features(load_config(args.config))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
