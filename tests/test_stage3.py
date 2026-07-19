from __future__ import annotations

import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

from genome_firewall.amrfinder_runner import (
    AnnotationItem,
    build_command,
    command_prefix,
    parse_version_output,
    valid_amrfinder_tsv,
)
from genome_firewall.config import load_config
from genome_firewall.dataset import Candidate, select_and_split
from genome_firewall.download_fastas import (
    DownloadItem,
    FastaStats,
    validate_expected_size,
    validate_fasta,
)
from genome_firewall.features import (
    build_vocabulary,
    encode_hit_sets,
    extract_element_symbols,
)


class StageThreeTests(unittest.TestCase):
    def test_feature_extraction_filters_scope_type_and_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "amrfinder.tsv"
            path.write_text(
                "Element symbol\tScope\tType\n"
                "blaTEM-1\tcore\tAMR\n"
                "blaTEM-1\tcore\tAMR\n"
                "plusOnly\tplus\tAMR\n"
                "stressGene\tcore\tSTRESS\n",
                encoding="utf-8",
            )
            self.assertEqual(extract_element_symbols(path), {"blaTEM-1"})

    def test_feature_vocabulary_is_fitted_on_training_hits_only(self) -> None:
        training_hits = [{"blaTEM-1", "sul2"}, {"sul2"}]
        feature_names = build_vocabulary(training_hits)
        self.assertEqual(feature_names, ["blaTEM-1", "sul2"])
        matrix, unseen = encode_hit_sets(
            [{"blaTEM-1"}, {"sul2", "futureGene"}],
            feature_names,
        )
        self.assertEqual(matrix.tolist(), [[1, 0], [0, 1]])
        self.assertEqual(unseen, [set(), {"futureGene"}])

    def test_amrfinder_version_parsing(self) -> None:
        output = """
Software version: 4.2.7
Database directory: /tmp/amrfinder/data/2026-05-15.1
Database version: 2026-05-15.1
"""
        self.assertEqual(
            parse_version_output(output),
            {
                "software_version": "4.2.7",
                "database_directory": "/tmp/amrfinder/data/2026-05-15.1",
                "database_version": "2026-05-15.1",
            },
        )

    def test_amrfinder_command_uses_core_mode_and_mutation_report(self) -> None:
        settings = {
            "organism": "Escherichia",
            "threads": 4,
            "write_mutation_all": True,
            "use_plus": False,
        }
        command = build_command(
            ["conda", "run", "-n", "amrfinder", "amrfinder"],
            settings,
            AnnotationItem("562.1", Path("sample.fna")),
            Path("sample.tsv"),
            Path("sample.mutations.tsv"),
        )
        self.assertIn("--mutation_all", command)
        self.assertNotIn("--plus", command)
        self.assertEqual(command[command.index("-O") + 1], "Escherichia")

    def test_container_runtime_resolves_amrfinder_directly(self) -> None:
        config = load_config("configs/mvp.yaml")
        with (
            patch.dict(
                "os.environ",
                {"GENOME_FIREWALL_AMRFINDER_EXECUTION": "direct"},
            ),
            patch(
                "genome_firewall.amrfinder_runner.shutil.which",
                return_value="/usr/local/bin/amrfinder",
            ),
        ):
            self.assertEqual(command_prefix(config), ["/usr/local/bin/amrfinder"])

    def test_amrfinder_tsv_validation_accepts_header_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.tsv"
            path.write_text("Element symbol\tScope\tType\n", encoding="utf-8")
            self.assertTrue(valid_amrfinder_tsv(path))

    def test_validate_fasta_counts_records_and_bases(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.fna"
            path.write_text(">one\nACGTN\n>two\nRYKM\n", encoding="ascii")
            stats = validate_fasta(path)
        self.assertEqual(stats.records, 2)
        self.assertEqual(stats.bases, 9)

    def test_validate_fasta_rejects_sequence_before_header(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.fna"
            path.write_text("ACGT\n", encoding="ascii")
            with self.assertRaises(ValueError):
                validate_fasta(path)

    def test_expected_size_rejects_truncated_download(self) -> None:
        item = DownloadItem(
            genome_id="562.1",
            url="https://example.invalid",
            destination=Path("sample.fna"),
            expected_bases=5_000_000,
        )
        with self.assertRaises(ValueError):
            validate_expected_size(item, FastaStats(records=25, bases=1_000_000))

    def test_selection_is_balanced_deterministic_and_disjoint(self) -> None:
        antibiotics = ["ciprofloxacin", "ampicillin"]
        groups = []
        candidates = []
        counter = 0
        for ciprofloxacin in ("resistant", "susceptible"):
            for ampicillin in ("resistant", "susceptible"):
                groups.append(
                    {
                        "ciprofloxacin": ciprofloxacin,
                        "ampicillin": ampicillin,
                        "count": 4,
                    }
                )
                for _ in range(8):
                    counter += 1
                    candidates.append(
                        Candidate(
                            genome_id=f"562.{counter}",
                            genome_name=f"Genome {counter}",
                            labels={
                                "ciprofloxacin": ciprofloxacin,
                                "ampicillin": ampicillin,
                            },
                            genome_length=5_000_000,
                            contigs=50,
                            checkm_completeness=99.0,
                            checkm_contamination=1.0,
                            mlst="",
                        )
                    )
        split_counts = {"train": 2, "calibration": 1, "test": 1}
        first = select_and_split(candidates, antibiotics, groups, split_counts, seed=42)
        second = select_and_split(candidates, antibiotics, groups, split_counts, seed=42)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 16)
        self.assertEqual(Counter(split for _, split in first.values())["train"], 8)
        self.assertEqual(Counter(split for _, split in first.values())["calibration"], 4)
        self.assertEqual(Counter(split for _, split in first.values())["test"], 4)


if __name__ == "__main__":
    unittest.main()
