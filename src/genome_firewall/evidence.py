"""Extract antibiotic-relevant AMRFinder evidence and statistical influences."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


def _tokens(value: Any) -> set[str]:
    text = str(value or "").upper().replace(";", "/").replace(",", "/")
    return {token.strip() for token in text.split("/") if token.strip() and token != "NA"}


def is_relevant_resistance_row(
    row: Mapping[str, Any],
    antibiotic_settings: Mapping[str, Any],
) -> bool:
    """Match known resistance rows through configured AMRFinder class mappings."""

    if str(row.get("Scope", "")).strip().lower() != "core":
        return False
    if str(row.get("Type", "")).strip().upper() != "AMR":
        return False
    configured_classes = {
        str(value).strip().upper() for value in antibiotic_settings.get("amrfinder_classes", [])
    }
    configured_subclasses = {
        str(value).strip().upper()
        for value in antibiotic_settings.get("amrfinder_subclasses", [])
    }
    return bool(
        configured_classes & _tokens(row.get("Class"))
        or configured_subclasses & _tokens(row.get("Subclass"))
    )


def _optional_float(value: Any) -> float | None:
    text = str(value or "").strip()
    if not text or text.upper() == "NA":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def collect_evidence(
    rows: Iterable[Mapping[str, Any]],
    antibiotic_settings: Mapping[str, Any],
    coefficients: Mapping[str, float],
    *,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Return the strongest directly mapped resistance rows."""

    relevant = [row for row in rows if is_relevant_resistance_row(row, antibiotic_settings)]
    relevant.sort(
        key=lambda row: (
            -abs(coefficients.get(str(row.get("Element symbol", "")), 0.0)),
            -(_optional_float(row.get("% Coverage of reference")) or 0.0),
            -(_optional_float(row.get("% Identity to reference")) or 0.0),
            str(row.get("Element symbol", "")),
        )
    )
    evidence: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in relevant:
        symbol = str(row.get("Element symbol", "")).strip()
        if not symbol or symbol in seen:
            continue
        seen.add(symbol)
        coefficient = coefficients.get(symbol)
        evidence.append(
            {
                "element_symbol": symbol,
                "element_name": str(row.get("Element name", "")).strip(),
                "type": str(row.get("Type", "")).strip(),
                "subtype": str(row.get("Subtype", "")).strip(),
                "class": str(row.get("Class", "")).strip(),
                "subclass": str(row.get("Subclass", "")).strip(),
                "method": str(row.get("Method", "")).strip(),
                "coverage": _optional_float(row.get("% Coverage of reference")),
                "identity": _optional_float(row.get("% Identity to reference")),
                "model_coefficient": coefficient,
            }
        )
        if len(evidence) == limit:
            break
    return evidence


def collect_model_contributions(
    detected_symbols: Iterable[str],
    coefficients: Mapping[str, float],
    *,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Rank detected model features without making causal claims."""

    ranked = sorted(
        (
            (symbol, coefficients[symbol])
            for symbol in set(detected_symbols)
            if symbol in coefficients
        ),
        key=lambda item: (-abs(item[1]), item[0]),
    )
    return [
        {
            "element_symbol": symbol,
            "coefficient": coefficient,
            "direction": "resistant" if coefficient > 0 else "susceptible",
        }
        for symbol, coefficient in ranked[:limit]
    ]
