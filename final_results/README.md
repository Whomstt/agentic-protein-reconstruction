# Final experiment results

The complete raw and derived output of the six reported runs: two organisms
(E. coli, yeast) × three digestion replica counts (5, 20, 100), 100 proteins
each, 600 samples in total. Every number in the write-up comes from these files.

All runs share one configuration apart from the swept axes: `gpt-5-mini` as the
agent (single-call mode, iteration 1 deterministic), `facebook/esm2_t6_8M_UR50D`
as the scoring PLM, 5 iterations per protein, seed 42.

| Folder | Organism | Replicas | Random-order APA | Agentic APA | Agentic tau | Agentic EM |
| --- | --- | --- | --- | --- | --- | --- |
| `ecoli_r100` | E. coli | 100 | 0.068 | 0.879 | 0.753 | 0.46 |
| `ecoli_r20` | E. coli | 20 | 0.120 | 0.734 | 0.624 | 0.29 |
| `ecoli_r5` | E. coli | 5 | 0.101 | 0.433 | 0.404 | 0.09 |
| `yeast_r100` | Yeast | 100 | 0.050 | 0.812 | 0.624 | 0.26 |
| `yeast_r20` | Yeast | 20 | 0.064 | 0.642 | 0.450 | 0.16 |
| `yeast_r5` | Yeast | 5 | 0.071 | 0.397 | 0.324 | 0.11 |

APA = adjacent pair accuracy, EM = exact match. Cross-run comparisons, replica
scaling and the organism analysis live in
[`_analysis/cross_run_report.md`](_analysis/cross_run_report.md).

## What is in each run folder

| File | What it is |
| --- | --- |
| `report.md` | The run's own report: paired Shuffled → Deterministic → Control → Agentic benchmark, oracle ceiling, validity-signal concordance, cost |
| `analysis_report.md` | Statistical layer, sections A–G: confidence intervals, paired tests with Holm correction, stratification, error taxonomy |
| `samples.jsonl.gz` | **The raw data.** One record per protein: every arm's reconstruction and metrics, plus the full per-iteration history (lever choices, LLM reasoning, junction stats, beam diagnostics) |
| `summary.json` | Run config snapshot and aggregate averages. The per-sample records it used to duplicate are in `samples.jsonl.gz` instead |
| `results.csv` | One row per protein — the same data as `samples.jsonl.gz`, flattened for spreadsheets |
| `summary.csv` | Arm × metric means with confidence intervals |
| `tables/`, `figures/` | Booktabs LaTeX and PNG figures generated from the above |

`_analysis/` holds the cross-run CSVs and figures; see its own `README.md` for
the column layout of each file.

## Regenerating any of this

Everything derived here comes from `samples.jsonl` plus the config snapshot —
no GPU, no model download, no network. To recompute from scratch, decompress and
point the rebuild at this folder:

```bash
gunzip -k final_results/*/samples.jsonl.gz
python -m evaluation.rebuild --all --results-root final_results
python -m evaluation.thesis_tables --all --results-root final_results
```

## Why the files are compressed

`samples.jsonl` is 5–7 MB per run uncompressed and gzips about 9×, which keeps
the whole published set at roughly 8 MB. The `samples` key was also removed from
`summary.json`, where it was a byte-identical duplicate of `samples.jsonl`.
Nothing else was filtered, truncated or rounded.

Live runs still write full-size output to `results/`, which is gitignored.
