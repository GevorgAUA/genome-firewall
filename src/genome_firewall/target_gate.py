"""Confirm configured antibiotic targets before allowing likely-to-work calls."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, Literal

TargetStatus = Literal["present", "absent", "unknown"]
DEFAULT_TARGET_GENES = {
    "ciprofloxacin": ["gyrA", "gyrB", "parC", "parE"],
    "ampicillin": ["ftsI"],
}


def element_gene(element_symbol: str) -> str:
    """Return the locus portion of an AMRFinder element symbol."""

    return element_symbol.strip().split("_", 1)[0]


def _row_gene(row: Mapping[str, Any]) -> str:
    for column in ("gene", "gene_symbol", "locus_tag", "Element symbol"):
        value = str(row.get(column, "")).strip()
        if value:
            return element_gene(value)
    return ""


def _as_rows(value: Any) -> list[Mapping[str, Any]]:
    """Accept record iterables or pandas-like DataFrames without importing pandas."""

    if value is None:
        return []
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        records = to_dict(orient="records")
        if isinstance(records, list):
            return records
    return list(value)


def assess_target(
    target_genes: Iterable[str],
    amrfinder_hits: Iterable[Mapping[str, Any]],
    mutation_report: Iterable[Mapping[str, Any]] | None = None,
    optional_annotation: Iterable[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Return target status plus the highest-priority evidence source."""

    configured = {gene.casefold(): gene for gene in target_genes}
    if not configured:
        return {"status": "unknown", "source": None, "found_genes": []}

    if optional_annotation is not None:
        annotation_rows = _as_rows(optional_annotation)
        found = sorted(
            {
                configured[gene.casefold()]
                for row in annotation_rows
                if (gene := _row_gene(row)).casefold() in configured
            }
        )
        return {
            "status": "present" if found else "absent",
            "source": "supplied_annotation",
            "found_genes": found,
        }

    for source, rows in (
        ("amrfinder_mutation_report", mutation_report),
        ("amrfinder_main_report", amrfinder_hits),
    ):
        if rows is None:
            continue
        found = sorted(
            {
                configured[gene.casefold()]
                for row in _as_rows(rows)
                if (gene := _row_gene(row)).casefold() in configured
            }
        )
        if found:
            return {"status": "present", "source": source, "found_genes": found}

    return {"status": "unknown", "source": None, "found_genes": []}


def check_target(
    antibiotic: str,
    amrfinder_hits: Iterable[Mapping[str, Any]],
    mutation_report: Iterable[Mapping[str, Any]] | None = None,
    optional_annotation: Iterable[Mapping[str, Any]] | None = None,
    *,
    antibiotic_settings: Mapping[str, Any] | None = None,
) -> TargetStatus:
    """Return only the target status for the public safety-gate contract."""

    if antibiotic_settings is None:
        target_genes = DEFAULT_TARGET_GENES.get(antibiotic.strip().lower())
        if target_genes is None:
            raise ValueError(f"Target settings are required for {antibiotic}")
        antibiotic_settings = {"target_genes_any": target_genes}
    assessment = assess_target(
        antibiotic_settings.get("target_genes_any", []),
        amrfinder_hits,
        mutation_report,
        optional_annotation,
    )
    return assessment["status"]
