"""Run AMRFinderPlus over the selected cohort with validation and caching."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from genome_firewall.config import LoadedConfig, load_config
from genome_firewall.download_fastas import validate_fasta

REQUIRED_AMRFINDER_COLUMNS = frozenset({"Element symbol", "Scope", "Type"})


@dataclass(frozen=True)
class AnnotationItem:
    genome_id: str
    fasta_path: Path


@dataclass(frozen=True)
class AnnotationPaths:
    output: Path
    mutations: Path
    stdout_log: Path
    stderr_log: Path


def valid_amrfinder_tsv(path: Path) -> bool:
    """Return whether a TSV has the current core columns, even with zero hits."""

    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle, delimiter="\t")
            header = next(reader)
    except (OSError, StopIteration, UnicodeError):
        return False
    return set(header) >= REQUIRED_AMRFINDER_COLUMNS


def parse_version_output(output: str) -> dict[str, str]:
    """Extract versions printed by ``amrfinder -l``."""

    patterns = {
        "software_version": r"Software version:\s*([^\s]+)",
        "database_directory": r"Database directory:\s*(.+)",
        "database_version": r"Database version:\s*([^\s]+)",
    }
    versions: dict[str, str] = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, output)
        if match:
            versions[key] = match.group(1).strip()
    return versions


def _conda_prefix(config: LoadedConfig) -> list[str]:
    settings = config.values["amrfinder"]
    raw_executable = os.path.expandvars(str(settings["conda_executable"]))
    conda_executable = Path(raw_executable).expanduser()
    if not conda_executable.is_file():
        raise FileNotFoundError(f"Conda executable not found: {conda_executable}")
    return [
        str(conda_executable),
        "run",
        "-n",
        str(settings["conda_environment"]),
        str(settings["executable"]),
    ]


def command_prefix(config: LoadedConfig) -> list[str]:
    """Resolve AMRFinder through Conda locally or directly in a container."""

    execution = os.environ.get(
        "GENOME_FIREWALL_AMRFINDER_EXECUTION",
        "conda",
    ).strip().casefold()
    if execution == "conda":
        return _conda_prefix(config)
    if execution != "direct":
        raise ValueError(
            "GENOME_FIREWALL_AMRFINDER_EXECUTION must be 'conda' or 'direct'"
        )

    executable = str(config.values["amrfinder"]["executable"])
    resolved = shutil.which(executable)
    if resolved is None:
        raise FileNotFoundError(f"AMRFinder executable not found in PATH: {executable}")
    return [resolved]


def _manifest_items(config: LoadedConfig) -> list[AnnotationItem]:
    manifest_path = config.resolve_path(config.values["paths"]["cohort_manifest"])
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Cohort manifest not found: {manifest_path}")
    items: list[AnnotationItem] = []
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            items.append(
                AnnotationItem(
                    genome_id=row["genome_id"],
                    fasta_path=config.resolve_path(row["fasta_path"]),
                )
            )
    if not items:
        raise ValueError(f"Cohort manifest is empty: {manifest_path}")
    return sorted(items, key=lambda item: item.genome_id)


def annotation_paths(output_dir: Path, genome_id: str) -> AnnotationPaths:
    logs_dir = output_dir / "logs"
    return AnnotationPaths(
        output=output_dir / f"{genome_id}.tsv",
        mutations=output_dir / f"{genome_id}.mutations.tsv",
        stdout_log=logs_dir / f"{genome_id}.stdout.log",
        stderr_log=logs_dir / f"{genome_id}.stderr.log",
    )


def build_command(
    prefix: list[str],
    settings: dict[str, Any],
    item: AnnotationItem,
    output: Path,
    mutations: Path,
) -> list[str]:
    command = [
        *prefix,
        "-n",
        str(item.fasta_path),
        "-O",
        str(settings["organism"]),
        "--name",
        item.genome_id,
        "--threads",
        str(settings["threads"]),
        "-o",
        str(output),
    ]
    if bool(settings["write_mutation_all"]):
        command.extend(["--mutation_all", str(mutations)])
    if bool(settings["use_plus"]):
        command.append("--plus")
    return command


def _write_log(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _run_one(
    config: LoadedConfig,
    prefix: list[str],
    item: AnnotationItem,
    force: bool,
    timeout_seconds: int,
    output_dir_override: Path | None = None,
) -> tuple[str, str]:
    settings = config.values["amrfinder"]
    output_dir = output_dir_override or config.resolve_path(
        config.values["paths"]["amrfinder_dir"]
    )
    paths = annotation_paths(output_dir, item.genome_id)
    if (
        not force
        and bool(settings["skip_valid_cached_outputs"])
        and valid_amrfinder_tsv(paths.output)
        and valid_amrfinder_tsv(paths.mutations)
    ):
        return item.genome_id, "cached"

    validate_fasta(item.fasta_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths.stdout_log.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = paths.output.with_suffix(".part.tsv")
    temporary_mutations = paths.mutations.with_suffix(".part.tsv")
    command = build_command(
        prefix,
        settings,
        item,
        temporary_output,
        temporary_mutations,
    )
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        _write_log(paths.stdout_log, completed.stdout)
        _write_log(paths.stderr_log, completed.stderr)
        if completed.returncode != 0:
            raise RuntimeError(
                f"AMRFinder returned {completed.returncode}; see {paths.stderr_log}"
            )
        if not valid_amrfinder_tsv(temporary_output):
            raise RuntimeError(f"Invalid AMRFinder output: {temporary_output}")
        if bool(settings["write_mutation_all"]) and not valid_amrfinder_tsv(
            temporary_mutations
        ):
            raise RuntimeError(f"Invalid mutation report: {temporary_mutations}")
        os.replace(temporary_output, paths.output)
        if bool(settings["write_mutation_all"]):
            os.replace(temporary_mutations, paths.mutations)
    except subprocess.TimeoutExpired as exc:
        _write_log(paths.stdout_log, exc.stdout or "")
        _write_log(paths.stderr_log, exc.stderr or "")
        raise RuntimeError(
            f"AMRFinder timed out after {timeout_seconds} seconds for {item.genome_id}"
        ) from exc
    finally:
        temporary_output.unlink(missing_ok=True)
        temporary_mutations.unlink(missing_ok=True)
    return item.genome_id, "processed"


def collect_versions(config: LoadedConfig, prefix: list[str]) -> dict[str, Any]:
    completed = subprocess.run(
        [*prefix, "-l"],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    combined = "\n".join((completed.stdout, completed.stderr))
    versions: dict[str, Any] = parse_version_output(combined)
    versions.update(
        {
            "organism": config.values["amrfinder"]["organism"],
            "use_plus": bool(config.values["amrfinder"]["use_plus"]),
            "recorded_at_utc": datetime.now(UTC).isoformat(),
        }
    )
    missing = {"software_version", "database_version"} - set(versions)
    if missing:
        raise RuntimeError(f"Could not parse AMRFinder version fields: {sorted(missing)}")
    return versions


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def run_batch(
    config: LoadedConfig,
    workers: int,
    limit: int | None,
    sample_id: str | None,
    force: bool,
    timeout_seconds: int,
) -> dict[str, Any]:
    prefix = command_prefix(config)
    items = _manifest_items(config)
    if sample_id is not None:
        items = [item for item in items if item.genome_id == sample_id]
        if not items:
            raise ValueError(f"Genome ID is not in the cohort manifest: {sample_id}")
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be at least 1")
        items = items[:limit]

    output_dir = config.resolve_path(config.values["paths"]["amrfinder_dir"])
    versions = collect_versions(config, prefix)
    _atomic_json(output_dir / "versions.json", versions)

    statuses: Counter[str] = Counter()
    failures: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _run_one,
                config,
                prefix,
                item,
                force,
                timeout_seconds,
            ): item
            for item in items
        }
        for future in as_completed(futures):
            item = futures[future]
            try:
                genome_id, status = future.result()
            except Exception as exc:
                failures[item.genome_id] = str(exc)
                print(f"FAILED {item.genome_id}: {exc}", flush=True)
            else:
                statuses[status] += 1
                print(f"{status.upper()} {genome_id}", flush=True)

    result: dict[str, Any] = {
        "requested": len(items),
        "processed": statuses["processed"],
        "cached": statuses["cached"],
        "failed": len(failures),
        "failures": failures,
        "software_version": versions["software_version"],
        "database_version": versions["database_version"],
        "organism": versions["organism"],
        "use_plus": versions["use_plus"],
    }
    _atomic_json(output_dir / "run_summary.json", result)
    if failures:
        failed_ids = ", ".join(sorted(failures))
        raise RuntimeError(f"AMRFinder failed for {len(failures)} genome(s): {failed_ids}")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--sample-id")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--timeout-seconds", type=int, default=1200)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.workers < 1 or args.workers > 4:
        raise SystemExit("--workers must be between 1 and 4")
    if args.timeout_seconds < 1:
        raise SystemExit("--timeout-seconds must be positive")
    result = run_batch(
        load_config(args.config),
        workers=args.workers,
        limit=args.limit,
        sample_id=args.sample_id,
        force=args.force,
        timeout_seconds=args.timeout_seconds,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
