"""Normalize BV-BRC metadata and select the fixed hackathon cohort."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from genome_firewall.config import LoadedConfig, load_config

csv.field_size_limit(10 * 1024 * 1024)


@dataclass(frozen=True)
class Candidate:
    genome_id: str
    genome_name: str
    labels: dict[str, str]
    genome_length: int
    contigs: int
    checkm_completeness: float | None
    checkm_contamination: float | None
    mlst: str


def _normal(value: str | None) -> str:
    return (value or "").strip().lower()


def _integer(value: str | None) -> int | None:
    try:
        return int(float((value or "").strip()))
    except ValueError:
        return None


def _number(value: str | None) -> float | None:
    try:
        return float((value or "").strip())
    except ValueError:
        return None


def _score(seed: int, purpose: str, genome_id: str) -> str:
    value = f"{seed}:{purpose}:{genome_id}".encode()
    return hashlib.sha256(value).hexdigest()


def _read_label_sets(
    path: Path,
    taxon_id: str,
    antibiotics: list[str],
    columns: dict[str, str],
) -> tuple[dict[tuple[str, str], set[str]], int]:
    labels: dict[tuple[str, str], set[str]] = defaultdict(set)
    matching_rows = 0
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            if (row.get(columns["taxon_id"]) or "").strip() != taxon_id:
                continue
            antibiotic = _normal(row.get(columns["antibiotic"]))
            if antibiotic not in antibiotics:
                continue
            matching_rows += 1
            genome_id = (row.get(columns["genome_id"]) or "").strip()
            phenotype = _normal(row.get(columns["phenotype"]))
            if genome_id and phenotype:
                labels[(genome_id, antibiotic)].add(phenotype)
    return labels, matching_rows


def _usable_labels(
    label_sets: dict[tuple[str, str], set[str]],
    antibiotics: list[str],
    resistant_label: str,
    susceptible_label: str,
) -> tuple[dict[str, dict[str, str]], Counter[str]]:
    by_genome: dict[str, dict[str, str]] = defaultdict(dict)
    exclusions: Counter[str] = Counter()
    allowed = {resistant_label, susceptible_label}

    for (genome_id, antibiotic), values in label_sets.items():
        if values == {resistant_label}:
            by_genome[genome_id][antibiotic] = resistant_label
        elif values == {susceptible_label}:
            by_genome[genome_id][antibiotic] = susceptible_label
        else:
            pattern = "+".join(sorted(values))
            exclusions[pattern or "blank"] += 1

    usable = {
        genome_id: labels
        for genome_id, labels in by_genome.items()
        if set(labels) == set(antibiotics) and set(labels.values()) <= allowed
    }
    return usable, exclusions


def _read_selected_rows(
    path: Path,
    wanted_ids: set[str],
    genome_id_column: str,
) -> dict[str, dict[str, str]]:
    selected: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            genome_id = (row.get(genome_id_column) or "").strip()
            if genome_id in wanted_ids:
                selected[genome_id] = row
    return selected


def _quality_candidate(
    genome_id: str,
    labels: dict[str, str],
    metadata: dict[str, str],
    summary: dict[str, str],
    columns: dict[str, str],
    quality: dict[str, Any],
) -> tuple[Candidate | None, list[str]]:
    genome_length = _integer(metadata.get(columns["genome_length"]))
    contigs = _integer(metadata.get(columns["contigs"]))
    completeness = _number(summary.get(columns["checkm_completeness"]))
    contamination = _number(summary.get(columns["checkm_contamination"]))
    reasons: list[str] = []

    if genome_length is None:
        reasons.append("missing_genome_length")
    elif not (
        int(quality["minimum_genome_length"])
        <= genome_length
        <= int(quality["maximum_genome_length"])
    ):
        reasons.append("genome_length")

    if contigs is None:
        reasons.append("missing_contigs")
    elif contigs > int(quality["maximum_contigs"]):
        reasons.append("contigs")

    minimum_completeness = float(quality["minimum_checkm_completeness_when_available"])
    if completeness is not None and completeness < minimum_completeness:
        reasons.append("checkm_completeness")

    maximum_contamination = float(quality["maximum_checkm_contamination_when_available"])
    if contamination is not None and contamination > maximum_contamination:
        reasons.append("checkm_contamination")

    if reasons or genome_length is None or contigs is None:
        return None, reasons

    return (
        Candidate(
            genome_id=genome_id,
            genome_name=(metadata.get(columns["genome_name"]) or "").strip().strip('"'),
            labels=labels,
            genome_length=genome_length,
            contigs=contigs,
            checkm_completeness=completeness,
            checkm_contamination=contamination,
            mlst=(metadata.get(columns["mlst"]) or "").strip(),
        ),
        [],
    )


def _group_key(candidate: Candidate, antibiotics: list[str]) -> tuple[str, ...]:
    return tuple(candidate.labels[antibiotic] for antibiotic in antibiotics)


def select_and_split(
    candidates: Iterable[Candidate],
    antibiotics: list[str],
    group_specs: list[dict[str, Any]],
    split_counts: dict[str, int],
    seed: int,
) -> dict[str, tuple[Candidate, str]]:
    """Select exact joint-label strata and assign reproducible splits."""

    pools: dict[tuple[str, ...], list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        pools[_group_key(candidate, antibiotics)].append(candidate)

    result: dict[str, tuple[Candidate, str]] = {}
    for spec in group_specs:
        key = tuple(str(spec[antibiotic]) for antibiotic in antibiotics)
        requested = int(spec["count"])
        pool = sorted(
            pools.get(key, []),
            key=lambda item: _score(seed, "cohort", item.genome_id),
        )
        if len(pool) < requested:
            raise ValueError(
                f"Joint phenotype group {key} needs {requested} genomes; only {len(pool)} qualify"
            )
        selected = sorted(
            pool[:requested],
            key=lambda item: _score(seed, "split", item.genome_id),
        )
        offset = 0
        for split_name in ("train", "calibration", "test"):
            count = int(split_counts[split_name])
            for candidate in selected[offset : offset + count]:
                if candidate.genome_id in result:
                    raise ValueError(f"Genome selected more than once: {candidate.genome_id}")
                result[candidate.genome_id] = (candidate, split_name)
            offset += count
        if offset != requested:
            raise ValueError(
                f"Split counts total {offset}, but group {key} requests {requested} genomes"
            )
    return result


def _atomic_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def build_dataset(config: LoadedConfig) -> dict[str, Any]:
    values = config.values
    dataset = values["dataset"]
    columns = dataset["columns"]
    antibiotics = [str(value) for value in values["scope"]["primary_antibiotics"]]
    taxon_id = str(values["scope"]["taxon_id"])
    resistant_label = str(dataset["labels"]["resistant"])
    susceptible_label = str(dataset["labels"]["susceptible"])

    amr_path = config.resolve_path(dataset["paths"]["amr"])
    metadata_path = config.resolve_path(dataset["paths"]["metadata"])
    summary_path = config.resolve_path(dataset["paths"]["summary"])
    for required in (amr_path, metadata_path, summary_path):
        if not required.is_file():
            raise FileNotFoundError(f"Required BV-BRC table not found: {required}")

    label_sets, matching_rows = _read_label_sets(
        amr_path,
        taxon_id,
        antibiotics,
        columns,
    )
    usable_labels, label_exclusions = _usable_labels(
        label_sets,
        antibiotics,
        resistant_label,
        susceptible_label,
    )
    wanted_ids = set(usable_labels)
    metadata_rows = _read_selected_rows(
        metadata_path,
        wanted_ids,
        columns["genome_id"],
    )
    summary_rows = _read_selected_rows(
        summary_path,
        wanted_ids,
        columns["genome_id"],
    )

    candidates: list[Candidate] = []
    quality_exclusions: Counter[str] = Counter()
    for genome_id, labels in usable_labels.items():
        metadata = metadata_rows.get(genome_id)
        summary = summary_rows.get(genome_id)
        if metadata is None or summary is None:
            quality_exclusions["missing_metadata_or_summary"] += 1
            continue
        candidate, reasons = _quality_candidate(
            genome_id,
            labels,
            metadata,
            summary,
            columns,
            values["cohort"]["quality"],
        )
        if candidate is None:
            quality_exclusions.update(reasons)
        else:
            candidates.append(candidate)

    selected = select_and_split(
        candidates,
        antibiotics,
        values["cohort"]["joint_phenotype_groups"],
        values["split"]["counts_per_joint_group"],
        int(values["split"]["seed"]),
    )
    ordered = sorted(selected.values(), key=lambda value: value[0].genome_id)
    fasta_dir = Path(values["paths"]["fasta_dir"])
    api_url = str(values["downloads"]["api_url"]).rstrip("/")

    manifest_rows: list[dict[str, Any]] = []
    normalized_rows: list[dict[str, Any]] = []
    for candidate, split_name in ordered:
        relative_fasta = (fasta_dir / f"{candidate.genome_id}.fna").as_posix()
        row: dict[str, Any] = {
            "genome_id": candidate.genome_id,
            "genome_name": candidate.genome_name,
            "species": values["scope"]["species_name"],
            "taxon_id": taxon_id,
            "ciprofloxacin_phenotype": candidate.labels["ciprofloxacin"],
            "ampicillin_phenotype": candidate.labels["ampicillin"],
            "split": split_name,
            "genome_length": candidate.genome_length,
            "contigs": candidate.contigs,
            "checkm_completeness": candidate.checkm_completeness,
            "checkm_contamination": candidate.checkm_contamination,
            "mlst": candidate.mlst,
            "fasta_path": relative_fasta,
            "fasta_url": (
                f"{api_url}/?eq(genome_id,{candidate.genome_id})&limit(1000)"
            ),
        }
        manifest_rows.append(row)
        for antibiotic in antibiotics:
            normalized_rows.append(
                {
                    "sample_id": candidate.genome_id,
                    "fasta_path": relative_fasta,
                    "species": values["scope"]["species_name"],
                    "antibiotic": antibiotic,
                    "phenotype": candidate.labels[antibiotic],
                    "split": split_name,
                    "group_id": "",
                }
            )

    manifest_path = config.resolve_path(values["paths"]["cohort_manifest"])
    normalized_path = config.resolve_path(values["paths"]["normalized_metadata"])
    genome_ids_path = manifest_path.parent / "genome_ids.txt"
    summary_output_path = manifest_path.parent / "cohort_summary.json"
    _atomic_csv(
        manifest_path,
        [
            "genome_id",
            "genome_name",
            "species",
            "taxon_id",
            "ciprofloxacin_phenotype",
            "ampicillin_phenotype",
            "split",
            "genome_length",
            "contigs",
            "checkm_completeness",
            "checkm_contamination",
            "mlst",
            "fasta_path",
            "fasta_url",
        ],
        manifest_rows,
    )
    _atomic_csv(
        normalized_path,
        [
            "sample_id",
            "fasta_path",
            "species",
            "antibiotic",
            "phenotype",
            "split",
            "group_id",
        ],
        normalized_rows,
    )
    _atomic_text(
        genome_ids_path,
        "".join(f"{row['genome_id']}\n" for row in manifest_rows),
    )

    joint_counts = Counter(
        (
            row["ciprofloxacin_phenotype"],
            row["ampicillin_phenotype"],
        )
        for row in manifest_rows
    )
    split_counts = Counter(row["split"] for row in manifest_rows)
    antibiotic_counts = {
        antibiotic: dict(
            Counter(row[f"{antibiotic}_phenotype"] for row in manifest_rows)
        )
        for antibiotic in antibiotics
    }
    result = {
        "source_matching_amr_rows": matching_rows,
        "genomes_with_usable_primary_labels": len(usable_labels),
        "quality_eligible_genomes": len(candidates),
        "selected_genomes": len(manifest_rows),
        "normalized_rows": len(normalized_rows),
        "split_counts": dict(split_counts),
        "antibiotic_counts": antibiotic_counts,
        "joint_counts": {"|".join(key): count for key, count in joint_counts.items()},
        "label_exclusions": dict(label_exclusions),
        "quality_exclusions": dict(quality_exclusions),
        "manifest_path": str(manifest_path),
        "normalized_metadata_path": str(normalized_path),
    }
    _atomic_text(summary_output_path, json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to the MVP YAML configuration")
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = build_dataset(load_config(args.config))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
