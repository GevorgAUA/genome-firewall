# Genome Firewall MVP Scope

Status: frozen for the hackathon baseline on 2026-07-19.

## Product scope

The MVP supports already assembled *Escherichia coli* genomes (`taxon_id=562`) and
uses AMRFinderPlus organism option `Escherichia`.

The primary antibiotics are:

1. ciprofloxacin, implemented first;
2. ampicillin, implemented after the ciprofloxacin path works end to end.

Trimethoprim/sulfamethoxazole is a stretch goal only. The MVP does not interpret MIC
values, identify species, assemble reads, or make clinical treatment recommendations.

## Label contract

Only an exact, unambiguous `resistant` or `susceptible` value is accepted. Blank,
intermediate, nonsusceptible, reduced-susceptibility, susceptible-dose-dependent, and
conflicting genome-antibiotic records are excluded.

The local BV-BRC data contains 5,068 *E. coli* genomes with usable labels for both
primary antibiotics. After the configured assembly-quality filter, 5,005 candidates
remain. Their joint phenotype counts are:

| Ciprofloxacin | Ampicillin | Eligible genomes |
| --- | --- | ---: |
| resistant | resistant | 1,077 |
| resistant | susceptible | 63 |
| susceptible | resistant | 1,565 |
| susceptible | susceptible | 2,300 |

## Cohort contract

Select 40 genomes from each joint phenotype group using a fixed random seed, producing
160 unique genomes. This gives both antibiotic models exactly 80 resistant and 80
susceptible examples while requiring only 160 FASTA downloads and AMRFinder runs.

Each joint group is split into 28 training, 6 calibration, and 6 test genomes. The
resulting complete cohort contains 112 training, 24 calibration, and 24 test genomes.
A genome must remain in one split across every antibiotic.

Eligible genomes must have:

- genome length from 4,000,000 through 6,500,000 bases;
- no more than 500 contigs;
- at least 95% CheckM completeness when that value is available;
- no more than 5% CheckM contamination when that value is available;
- an available BV-BRC FASTA download.

An NCBI assembly accession is not required because BV-BRC sequence downloads use the
BV-BRC `genome_id` directly.

## Decision contract

Models target at least 95% precision on the calibration split. Calls that do not meet
the learned fail/work thresholds return `no_call`. Report called-sample accuracy and
the no-call rate together; do not describe a high-confidence statistical association
as biological causation.

## Stage 2 acceptance record

- Species and AMRFinder organism are fixed.
- Two primary antibiotics and one stretch antibiotic are fixed.
- Included and excluded phenotype labels are fixed.
- Cohort size, balancing rule, quality filter, and splits are fixed.
- Model baseline and conservative decision target are fixed.
- Machine-readable settings are stored in `configs/mvp.yaml`.

The next stage is to generate the deterministic cohort manifest, verify remote FASTA
availability, and download only the 160 selected genomes.

Genome FASTAs are retrieved from the certificate-valid BV-BRC HTTPS Data API using
the `application/dna+fasta` response type. The downloader does not disable TLS
certificate verification.
