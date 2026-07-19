"""Download and validate the FASTAs named by the fixed cohort manifest."""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from genome_firewall.config import LoadedConfig, load_config


@dataclass(frozen=True)
class FastaStats:
    records: int
    bases: int


@dataclass(frozen=True)
class DownloadItem:
    genome_id: str
    url: str
    destination: Path
    expected_bases: int


IUPAC_DNA = frozenset("ACGTRYSWKMBDHVN.-")


def validate_fasta(path: Path) -> FastaStats:
    """Validate FASTA structure and accepted IUPAC nucleotide characters."""

    records = 0
    bases = 0
    in_record = False
    with path.open("r", encoding="ascii") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if len(line) == 1:
                    raise ValueError(f"Empty FASTA header at line {line_number}: {path}")
                records += 1
                in_record = True
                continue
            if not in_record:
                raise ValueError(f"Sequence precedes first FASTA header: {path}")
            sequence = line.upper()
            invalid = set(sequence) - IUPAC_DNA
            if invalid:
                symbols = "".join(sorted(invalid))
                raise ValueError(
                    f"Invalid FASTA symbols {symbols!r} at line {line_number}: {path}"
                )
            bases += len(sequence.replace("-", "").replace(".", ""))
    if records == 0 or bases == 0:
        raise ValueError(f"FASTA contains no sequence records: {path}")
    return FastaStats(records=records, bases=bases)


def validate_expected_size(item: DownloadItem, stats: FastaStats) -> None:
    """Reject truncated or unexpectedly different genome downloads."""

    tolerance = max(1_000, int(item.expected_bases * 0.01))
    difference = abs(stats.bases - item.expected_bases)
    if difference > tolerance:
        raise ValueError(
            f"Genome {item.genome_id} contains {stats.bases} FASTA bases; "
            f"expected approximately {item.expected_bases}"
        )


def _manifest_items(config: LoadedConfig) -> list[DownloadItem]:
    manifest_path = config.resolve_path(config.values["paths"]["cohort_manifest"])
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Cohort manifest not found: {manifest_path}. Run the dataset stage first."
        )
    items: list[DownloadItem] = []
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            items.append(
                DownloadItem(
                    genome_id=row["genome_id"],
                    url=row["fasta_url"],
                    destination=config.resolve_path(row["fasta_path"]),
                    expected_bases=int(row["genome_length"]),
                )
            )
    if not items:
        raise ValueError(f"Cohort manifest is empty: {manifest_path}")
    return items


def _curl_base(config: LoadedConfig) -> list[str]:
    downloads = config.values["downloads"]
    return [
        "curl",
        "--fail",
        "--show-error",
        "--silent",
        "--location",
        "--header",
        f"Accept: {downloads['accept']}",
        "--connect-timeout",
        "30",
        "--retry",
        str(downloads["retries"]),
        "--retry-delay",
        "2",
    ]


def check_remote(config: LoadedConfig, item: DownloadItem) -> str:
    command = [*_curl_base(config), "--head", item.url]
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
    return item.genome_id


def download_one(
    config: LoadedConfig,
    item: DownloadItem,
) -> tuple[str, str, FastaStats]:
    item.destination.parent.mkdir(parents=True, exist_ok=True)
    if item.destination.is_file():
        try:
            stats = validate_fasta(item.destination)
            validate_expected_size(item, stats)
        except (OSError, UnicodeError, ValueError):
            pass
        else:
            return item.genome_id, "cached", stats

    partial = item.destination.with_suffix(item.destination.suffix + ".part")
    command = [*_curl_base(config), "--output", str(partial), item.url]
    try:
        subprocess.run(command, check=True)
        stats = validate_fasta(partial)
        validate_expected_size(item, stats)
        os.replace(partial, item.destination)
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    return item.genome_id, "downloaded", stats


def run_downloads(
    config: LoadedConfig,
    workers: int,
    limit: int | None,
    check_only: bool,
) -> dict[str, Any]:
    items = _manifest_items(config)
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be at least 1")
        items = items[:limit]

    completed: list[tuple[str, str, FastaStats]] = []
    failures: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                check_remote if check_only else download_one,
                config,
                item,
            ): item
            for item in items
        }
        for future in as_completed(futures):
            item = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                failures[item.genome_id] = str(exc)
                print(f"FAILED {item.genome_id}: {exc}")
            else:
                if check_only:
                    print(f"AVAILABLE {result}")
                else:
                    completed.append(result)
                    genome_id, status, stats = result
                    print(
                        f"{status.upper()} {genome_id}: "
                        f"{stats.records} records, {stats.bases} bases"
                    )

    if failures:
        failed = ", ".join(sorted(failures))
        raise RuntimeError(f"{len(failures)} FASTA operation(s) failed: {failed}")

    return {
        "checked": len(items) if check_only else 0,
        "downloaded": sum(status == "downloaded" for _, status, _ in completed),
        "cached": sum(status == "cached" for _, status, _ in completed),
        "total": len(items),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to the MVP YAML configuration")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Verify remote files without downloading their contents",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.workers < 1 or args.workers > 8:
        raise SystemExit("--workers must be between 1 and 8")
    result = run_downloads(
        load_config(args.config),
        workers=args.workers,
        limit=args.limit,
        check_only=args.check_only,
    )
    print(result)


if __name__ == "__main__":
    main()
