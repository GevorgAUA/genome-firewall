# Data included in this repository

Only three public BV-BRC *Escherichia coli* demonstration genomes are committed:

| Genome ID | Demo purpose |
|---|---|
| `562.100008` | Mixed ciprofloxacin work / ampicillin fail result |
| `562.45637` | Ampicillin probability-zone no-call |
| `562.100190` | Likely-to-work results for both antibiotics |

Each genome has one assembled nucleotide FASTA under `raw/fasta/` and two cached
AMRFinderPlus reports under `amrfinder/`. These reports power the instant UI demos.

The full 160-genome cohort, normalized metadata, feature matrices, and approximately
821 MB of source downloads are deliberately excluded. They can be regenerated using the
pipeline commands in the repository root README and the public BV-BRC source.

## Attribution

Please cite:

- Olson RD et al. Introducing the Bacterial and Viral Bioinformatics Resource Center
  (BV-BRC). *Nucleic Acids Research*. 2023;51(D1):D678–D689.
  <https://doi.org/10.1093/nar/gkac1003>
- Feldgarden M et al. AMRFinderPlus and the Reference Gene Catalog facilitate
  examination of the genomic links among antimicrobial resistance, stress response,
  and virulence. *Scientific Reports*. 2021;11:12728.
  <https://doi.org/10.1038/s41598-021-91456-0>
