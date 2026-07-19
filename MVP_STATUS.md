# Genome Firewall MVP status

The frozen hackathon baseline is complete for assembled *Escherichia coli* FASTA input.

- [x] Deterministic balanced 160-genome cohort
- [x] Validated FASTA downloads and cached AMRFinder outputs
- [x] Train-only 115-feature AMR vocabulary
- [x] Independent ciprofloxacin and ampicillin logistic models
- [x] Held-out Platt calibration and three-way decision thresholds
- [x] Blind 24-genome-per-antibiotic test evaluation
- [x] Target-presence gate before every `likely_to_work` decision
- [x] Direct AMRFinder evidence and ranked statistical influences
- [x] Structured failure and `no_call` reasons
- [x] End-to-end new-FASTA inference tested successfully
- [x] Streamlit upload interface, instant demo, and JSON download
- [x] Mandatory laboratory-confirmation warning

## Frozen blind-test snapshot

| Antibiotic | Balanced accuracy | AUROC | Called accuracy | No-call rate |
|---|---:|---:|---:|---:|
| Ciprofloxacin | 91.7% | 97.9% | 91.7% | 0.0% |
| Ampicillin | 91.7% | 95.1% | 94.1% | 29.2% |

These results use only 24 test genomes per antibiotic and are not clinical validation.
Models and thresholds must not be retuned using these test genomes.

## Demo launch

From WSL in the project directory:

```bash
$HOME/.local/bin/uv run --locked streamlit run app.py
```

Then open <http://localhost:8501> and select **Run instant demo**, or upload an assembled
*E. coli* nucleotide FASTA.
