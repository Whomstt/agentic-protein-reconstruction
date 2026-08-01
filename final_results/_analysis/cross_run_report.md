# Cross-Run Statistical Report

_Generated 2026-08-01 18:03 from 6 runs, 600 samples total. All values recomputed from stored per-sample data._

## Run matrix

| Organism | Replicas | n | Random Order APA | LLM-Guided APA | LLM-Guided tau | LLM-Guided EM |
| --- | --- | --- | --- | --- | --- | --- |
| E. coli | 100 | 100 | 0.0676 | 0.8791 | 0.7532 | 0.4600 |
| E. coli | 20 | 100 | 0.1199 | 0.7338 | 0.6242 | 0.2900 |
| E. coli | 5 | 100 | 0.1012 | 0.4328 | 0.4043 | 0.0900 |
| Yeast | 100 | 100 | 0.0502 | 0.8118 | 0.6241 | 0.2600 |
| Yeast | 20 | 100 | 0.0640 | 0.6417 | 0.4500 | 0.1600 |
| Yeast | 5 | 100 | 0.0712 | 0.3972 | 0.3243 | 0.1100 |

## C. Replica Scaling

More digestion replicas mean more adjacencies the overlap graph can confirm outright — near-ground-truth structure the search gets for free.

| Organism | Replicas | Confirmed adjacencies | APA | Kendall tau |
| --- | --- | --- | --- | --- |
| E. coli | 5 | 4.76 | 0.4328 | 0.4043 |
| E. coli | 20 | 12.95 | 0.7338 | 0.6242 |
| E. coli | 100 | 18.20 | 0.8791 | 0.7532 |
| Yeast | 5 | 9.97 | 0.3972 | 0.3243 |
| Yeast | 20 | 23.24 | 0.6417 | 0.4500 |
| Yeast | 100 | 30.75 | 0.8118 | 0.6241 |

## E. coli vs Yeast, conditioned on fragment count

A raw organism gap can be a protein-length artifact: if one organism's proteins are simply cut into more fragments, it will look harder without any biological difference. Conditioning on fragment count is what separates the two.

| Fragments | E. coli | Yeast |
| --- | --- | --- |
| 2-4 | 1.0000 (n=14, len 53) | 1.0000 (n=9, len 93) |
| 5-9 | 0.7915 (n=58, len 117) | 0.7543 (n=43, len 138) |
| 10-19 | 0.6593 (n=125, len 267) | 0.6701 (n=62, len 198) |
| 20-49 | 0.6010 (n=82, len 412) | 0.5793 (n=127, len 456) |
| 50+ | 0.6480 (n=20, len 861) | 0.4858 (n=58, len 1026) |

## Provenance

- `all_runs_overview.csv` (6 rows) - one row per run, headline numbers
- `all_runs_summary.csv` (180 rows) - arm x metric means with CIs
- `all_runs_results.csv` (600 rows) - one row per sample
- `all_runs_tests.csv` (72 rows) - paired tests with Holm-adjusted p
- `all_runs_stratified.csv` (900 rows) - by fragment count, with lift
- `all_runs_taxonomy.csv` (42 rows) - error-mode composition
- See `README.md` in this folder for what each file is for.
- No GPU, model loading or network access.
