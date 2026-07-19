from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from genome_firewall.evidence import (
    collect_evidence,
    collect_model_contributions,
    is_relevant_resistance_row,
)
from genome_firewall.predict import finalize_decision, predict_genome
from genome_firewall.target_gate import assess_target, check_target


class InferenceTests(unittest.TestCase):
    @property
    def config_path(self) -> Path:
        return Path(__file__).resolve().parents[1] / "configs" / "mvp.yaml"

    def test_mutation_report_confirms_target_locus(self) -> None:
        assessment = assess_target(
            ["ftsI"],
            [],
            [{"Element symbol": "ftsI_A498A"}],
        )
        self.assertEqual(assessment["status"], "present")
        self.assertEqual(assessment["source"], "amrfinder_mutation_report")
        self.assertEqual(assessment["found_genes"], ["ftsI"])

    def test_comprehensive_annotation_can_establish_absence(self) -> None:
        assessment = assess_target(
            ["ftsI"],
            [],
            [{"Element symbol": "ftsI_A498A"}],
            optional_annotation=[{"gene": "other_gene"}],
        )
        self.assertEqual(assessment["status"], "absent")
        self.assertEqual(assessment["source"], "supplied_annotation")

    def test_target_is_unknown_without_positive_or_comprehensive_evidence(self) -> None:
        self.assertEqual(
            check_target("ciprofloxacin", [], []),
            "unknown",
        )

    def test_relevant_evidence_uses_class_and_subclass_tokens(self) -> None:
        settings = {
            "amrfinder_classes": ["QUINOLONE"],
            "amrfinder_subclasses": ["QUINOLONE"],
        }
        row = {
            "Element symbol": "marR_S3N",
            "Scope": "core",
            "Type": "AMR",
            "Class": "MULTIDRUG",
            "Subclass": "AMPICILLIN/QUINOLONE/TETRACYCLINE",
        }
        self.assertTrue(is_relevant_resistance_row(row, settings))
        row["Subclass"] = "SULFONAMIDE"
        self.assertFalse(is_relevant_resistance_row(row, settings))

    def test_evidence_is_deduplicated_and_ranked_by_model_influence(self) -> None:
        settings = {"amrfinder_classes": ["BETA-LACTAM"], "amrfinder_subclasses": []}
        rows = [
            {
                "Element symbol": "blaA",
                "Element name": "A",
                "Scope": "core",
                "Type": "AMR",
                "Class": "BETA-LACTAM",
                "% Coverage of reference": "100",
                "% Identity to reference": "99",
            },
            {
                "Element symbol": "blaB",
                "Element name": "B",
                "Scope": "core",
                "Type": "AMR",
                "Class": "BETA-LACTAM",
                "% Coverage of reference": "100",
                "% Identity to reference": "99",
            },
        ]
        evidence = collect_evidence(rows, settings, {"blaA": 0.5, "blaB": 2.0})
        self.assertEqual([item["element_symbol"] for item in evidence], ["blaB", "blaA"])

    def test_model_contributions_are_labeled_as_direction_not_causation(self) -> None:
        contributions = collect_model_contributions(
            ["geneA", "geneB"],
            {"geneA": -2.0, "geneB": 0.5},
        )
        self.assertEqual(contributions[0]["element_symbol"], "geneA")
        self.assertEqual(contributions[0]["direction"], "susceptible")

    def test_target_gate_blocks_only_likely_to_work(self) -> None:
        thresholds = {"work": {"threshold": 0.2}, "fail": {"threshold": 0.8}}
        self.assertEqual(
            finalize_decision(0.1, thresholds, "unknown")[:2],
            ("no_call", "target_unknown"),
        )
        self.assertEqual(
            finalize_decision(0.9, thresholds, "unknown")[:2],
            ("likely_to_fail", None),
        )

    def test_uncertain_probability_remains_no_call_even_when_target_present(self) -> None:
        thresholds = {"work": {"threshold": 0.2}, "fail": {"threshold": 0.8}}
        self.assertEqual(
            finalize_decision(0.5, thresholds, "present")[:2],
            ("no_call", "probability_in_uncertain_zone"),
        )

    def test_invalid_fasta_returns_structured_no_call(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.fna"
            path.write_text("not a fasta\n", encoding="ascii")
            result = predict_genome(path, config_path=self.config_path)
        self.assertEqual(len(result), 2)
        self.assertTrue(all(item["prediction"] == "no_call" for item in result))
        self.assertTrue(all(item["no_call_reason"] == "invalid_fasta" for item in result))

    def test_unsupported_antibiotic_returns_structured_no_call(self) -> None:
        result = predict_genome(
            "missing.fna",
            "vancomycin",
            config_path=self.config_path,
        )
        self.assertEqual(result[0]["antibiotic"], "vancomycin")
        self.assertEqual(result[0]["no_call_reason"], "unsupported_antibiotic")


if __name__ == "__main__":
    unittest.main()
