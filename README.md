---
title: Genome Firewall
emoji: "🧬"
colorFrom: red
colorTo: blue
sdk: docker
app_port: 7860
suggested_hardware: cpu-basic
license: mit
---

# Genome Firewall

[![CI](https://github.com/GevorgAUA/genome-firewall/actions/workflows/ci.yml/badge.svg)](https://github.com/GevorgAUA/genome-firewall/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11–3.12](https://img.shields.io/badge/Python-3.11–3.12-3776AB.svg)](pyproject.toml)

Genome Firewall is an interpretable, uncertainty-aware research prototype that predicts
antibiotic-resistance phenotypes from an assembled *Escherichia coli* genome FASTA.

**[Open the live Genome Firewall demo](https://huggingface.co/spaces/GevorgAUA/genome-firewall)**

> **Research prototype only. Not for clinical use. Confirm every prediction with
> validated laboratory antimicrobial susceptibility testing.**

## What it does

```text
Assembled E. coli FASTA
  → AMRFinderPlus genes and mutations
  → fixed train-only AMR feature vocabulary
  → one calibrated logistic model per antibiotic
  → drug-target safety gate
  → likely_to_work / likely_to_fail / no_call
  → inspectable evidence report
```

The MVP supports ciprofloxacin and ampicillin. It does not assemble sequencing reads,
identify species, recommend patient treatment, modify genomes, or replace laboratory
testing.

## Blind holdout result

Models and decision thresholds were frozen before evaluation on 24 held-out genomes per
antibiotic.

| Antibiotic | Balanced accuracy | AUROC | Called accuracy | No-call rate |
|---|---:|---:|---:|---:|
| Ciprofloxacin | 91.7% | 97.9% | 91.7% | 0.0% |
| Ampicillin | 91.7% | 95.1% | 94.1% | 29.2% |

This is a small hackathon evaluation, not clinical validation. Full metrics, individual
predictions, reliability plots, and small-sample confidence intervals are in
[`reports/`](reports/).

## Instant local demo

Requirements:

- Windows with WSL2 Ubuntu, or a Linux host;
- Python 3.11 or 3.12;
- [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync --all-extras --dev --locked
uv run --locked streamlit run app.py
```

Open <http://localhost:8501>. The three instant demos use committed, precomputed
AMRFinder reports and do not require AMRFinder to be installed.

The interface demonstrates:

- a mixed likely-to-work / likely-to-fail result;
- an explicit safety no-call;
- likely-to-work calls for both supported antibiotics;
- confirmed target loci, known AMR evidence, statistical influences, and JSON export.

## Upload a FASTA

Upload an assembled *E. coli* nucleotide FASTA ending in `.fna`, `.fa`, `.fasta`, or
`.fas`, up to 25 MB. Raw FASTQ reads and non-*E. coli* genomes are unsupported.

New uploads require AMRFinderPlus 4.2.7 with the database recorded in
[`data/amrfinder/versions.json`](data/amrfinder/versions.json). The local MVP configuration
expects the Conda environment created during setup at `$HOME/miniforge3/envs/amrfinder`.

```bash
uv run --locked genome-firewall predict --fasta /path/to/genome.fna
```

Use `--antibiotic ciprofloxacin` or `--antibiotic ampicillin` to request one model.

## Reproduce the pipeline

The full local BV-BRC download is intentionally excluded from GitHub. After placing the
three source TSVs under `data/`, run:

```bash
uv run --locked python -m genome_firewall.dataset --config configs/mvp.yaml
uv run --locked python -m genome_firewall.download_fastas --config configs/mvp.yaml --workers 4
uv run --locked python -m genome_firewall.amrfinder_runner --config configs/mvp.yaml --workers 1
uv run --locked python -m genome_firewall.features --config configs/mvp.yaml
uv run --locked python -m genome_firewall.train --config configs/mvp.yaml
uv run --locked python -m genome_firewall.evaluate --config configs/mvp.yaml
```

Never retrain or retune using the already unblinded test genomes.

## Development

```bash
uv sync --all-extras --dev --locked
uv run --locked ruff check app.py src tests
uv run --locked pytest -q
```

See [`MVP_STATUS.md`](MVP_STATUS.md) for the acceptance checklist and
[`MVP_SCOPE.md`](MVP_SCOPE.md) for the frozen cohort contract.

## Data and software attribution

The three demonstration genomes and phenotype source data come from the public
[Bacterial and Viral Bioinformatics Resource Center (BV-BRC)](https://www.bv-brc.org/).
BV-BRC requests citation of Olson et al., *Nucleic Acids Research* (2023),
[doi:10.1093/nar/gkac1003](https://doi.org/10.1093/nar/gkac1003).

AMR annotations use [NCBI AMRFinderPlus](https://github.com/ncbi/amr). Please cite
Feldgarden et al., *Scientific Reports* (2021),
[doi:10.1038/s41598-021-91456-0](https://doi.org/10.1038/s41598-021-91456-0).

## License

Code is released under the [MIT License](LICENSE). Public biological source data and
third-party software remain subject to their respective attribution terms.
