# Agentic Evaluation (Yeast, esm_small, r100)

## How to Read This Report
**Run type: Agentic (single_call).** Each metric compares:

- **Shuffled Baseline** — a random fragment ordering. The floor, not a method.
- **Deterministic (config defaults)** — the non-agentic baseline: iteration 1 run with the fixed `search.default_levers` and **no LLM call**. The agent refines from it.
- **Control (non-LLM)** — the matched-budget control arm: the SAME iteration budget, SAME fixed tool pipeline and SAME best-validity selection as the agentic arm, but the five levers are chosen by a non-LLM policy (random/grid) instead of the LLM, and it runs paired on the same protein with 0 LLM calls. **`Δ Agentic − Control` is the isolated value of the LLM's reasoning** — it separates "the agent reasons well" from "trying several candidates and keeping the best-validity one helps."
- **Agentic Best** — the agent's result: iterations 2+ are LLM lever choices, and the kept candidate is the best-validity one across all iterations (subject to `search.improvement_margin`). Since iteration 1 (the deterministic baseline) is in the candidate set, read the **true-metric** columns for the real "does the agent help?" answer.
- **Oracle (ceiling)** — for each metric, the best value achievable by selecting among the candidates the agent actually generated. Not a method (it peeks at the ground truth); the **Oracle − Agentic** gap is what the imperfect (~57–61%) validity concordance leaves on the table — reachable by better selection alone, no new candidate.

## Run Overview
- Samples evaluated: 100
- Avg junctions pruned: 5.0%
- Exact matches: 26/100
- Result folder: `140726_171539_agentic`
- Total run duration: 2h 13m 20s
- Avg duration per sample: 1m 20s

## Configuration
| Setting | Value |
| --- | --- |
| Run Method | agentic |
| Calling Mode | single_call |
| Device | cuda |
| Seed | 42 |
| Dataset | Yeast |
| Test Samples | 100 |
| Replica Count | 100 |
| Missed Cleavage Ratio | 0.3 |
| MLM Model | facebook/esm2_t6_8M_UR50D |
| MLM Type | esm |
| MLM Batch Size | 64 |
| MLM Max Length | 1024 |
| Beam Width | 25 |
| Junction Window | 1 |
| Validity Model | facebook/esm2_t6_8M_UR50D |
| Max Iterations | 5 |
| Iteration 1 Mode | Deterministic — config defaults (no LLM call) |
| Early Stop Patience | 5 |
| LLM Model | gpt-5-mini |
| LLM Sampling | seed=42, reasoning_effort=low, verbosity=low |

## Benchmark: Shuffled Baseline vs. Deterministic vs. Agentic vs. Control
| Metric | Shuffled Baseline | Deterministic (config defaults, no LLM) | Control (random, no LLM) | Agentic Best (iteratively selected) | Oracle (ceiling) | Δ Agentic − Deterministic | Δ Agentic − Control | Direction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Exact Match | 0.0000 | 0.1600 | 0.2600 | 0.2600 | 0.3500 | +0.1000 | +0.0000 | better |
| Sequence Similarity | 0.2128 | 0.5095 | 0.5485 | 0.5474 | 0.6474 | +0.0379 | -0.0011 | better |
| Adjacent Pair Accuracy | 0.0502 | 0.7034 | 0.7876 | 0.8118 | 0.8546 | +0.1083 | +0.0241 | better |
| Longest Correct Run | 0.0937 | 0.4479 | 0.5128 | 0.5297 | 0.6226 | +0.0818 | +0.0169 | better |
| Kendall Tau | -0.0308 | 0.5005 | 0.5984 | 0.6241 | 0.7328 | +0.1236 | +0.0257 | better |

![Metric comparison](metric_comparison.svg)

### Selection Ceiling (Oracle)
The Oracle column is the best true-metric value reachable by selecting among the candidates the agent already generated (it peeks at ground truth, so it is a ceiling, not a method). The gap below is quality the run left on the table purely to imperfect validity selection — reachable with better selection alone, no new candidate generated. A large gap says the bottleneck is the selection signal, not the search.

| Metric | Agentic Best (iteratively selected) | Oracle (best generated) | Gap (Oracle − Agentic) |
| --- | --- | --- | --- |
| Exact Match | 0.2600 | 0.3500 | +0.0900 |
| Sequence Similarity | 0.5474 | 0.6474 | +0.1000 |
| Adjacent Pair Accuracy | 0.8118 | 0.8546 | +0.0428 |
| Longest Correct Run | 0.5297 | 0.6226 | +0.0929 |
| Kendall Tau | 0.6241 | 0.7328 | +0.1087 |

## Agentic vs. Deterministic (paired, per sample, n=100)
Per-sample gain of the agentic result over the deterministic best-fixed baseline (Agentic − Deterministic on the same protein). Mean is the average improvement; std dev shows how consistently the agent helps vs. swings the other way on individual samples. For a significance claim on n samples, run a paired Wilcoxon signed-rank test on these per-sample gains.

| Metric | Mean Gain | Std Dev | Min Gain | Max Gain |
| --- | --- | --- | --- | --- |
| Exact Match | +0.1000 | 0.3000 | +0.0000 | +1.0000 |
| Sequence Similarity | +0.0379 | 0.1650 | -0.6052 | +0.5375 |
| Adjacent Pair Accuracy | +0.1083 | 0.1455 | -0.0769 | +0.6667 |
| Longest Correct Run | +0.0818 | 0.1941 | -0.2903 | +0.7500 |
| Kendall Tau | +0.1236 | 0.2958 | -0.3556 | +1.0606 |
| Best Validity Score (junction+overlap blend, lower=better) | -1.9851 | 2.0232 | -9.4542 | +0.0000 |

## Agentic vs. Control (paired, matched budget, n=100)
**The isolated value of the LLM's reasoning.** Per-sample gain of the agentic arm over the non-LLM control arm (Agentic − Control on the same protein), where both arms ran the same iteration budget, the same fixed tool pipeline and the same best-validity selection — only the lever *source* differed (LLM vs a random policy). A plain Agentic − Deterministic gain conflates 'the agent reasons well' with 'trying several candidates and keeping the best helps'; this comparison holds the budget and selection fixed, so a positive, consistent gain here is attributable to the LLM. Run a paired Wilcoxon signed-rank test on these per-sample gains for the significance claim.

| Metric | Mean Gain | Std Dev | Min Gain | Max Gain |
| --- | --- | --- | --- | --- |
| Exact Match | +0.0000 | 0.0000 | +0.0000 | +0.0000 |
| Sequence Similarity | -0.0011 | 0.1213 | -0.6737 | +0.3366 |
| Adjacent Pair Accuracy | +0.0241 | 0.0710 | -0.1429 | +0.2857 |
| Longest Correct Run | +0.0169 | 0.0858 | -0.2121 | +0.4500 |
| Kendall Tau | +0.0257 | 0.1727 | -0.4067 | +0.9286 |
| Best Validity Score (lower=better; negative = agentic more plausible) | -0.0243 | 0.7804 | -2.0991 | +2.6621 |

## Distribution Summary (n=100 samples)
The at-a-glance view for larger runs — read this before the per-sample table.

| Metric | Mean | Std Dev | Min | Median | Max |
| --- | --- | --- | --- | --- | --- |
| Exact Match | 0.2600 | 0.4386 | 0.0000 | 0.0000 | 1.0000 |
| Sequence Similarity | 0.5474 | 0.3602 | 0.0005 | 0.5554 | 1.0000 |
| Adjacent Pair Accuracy | 0.8118 | 0.1419 | 0.3333 | 0.8000 | 1.0000 |
| Longest Correct Run | 0.5297 | 0.3154 | 0.0824 | 0.4545 | 1.0000 |
| Kendall Tau | 0.6241 | 0.3267 | -0.1515 | 0.6560 | 1.0000 |
| Best Validity Score (junction+overlap blend, lower=better) | 14.1427 | 5.7244 | 1.0000 | 15.9221 | 23.2077 |

## Validity Signal Concordance
Whether the validity score used to *select* the winning candidate actually tracks true reconstruction quality, measured within each sample across the iterations it tried. 0.50 = no better than chance at picking the better of two candidates; higher is better. This is the trust check for the selection signal — if it is near 0.50, a better candidate the agent generated would not reliably be the one kept.

| Quality metric compared against | Concordance | Comparable pairs |
| --- | --- | --- |
| Kendall Tau | 0.627 | 113925 |
| Adjacent Pair Accuracy | 0.656 | 113438 |

## Cost, Efficiency & Completion
The non-quality axis the agentic approach must also justify itself on: LLM calls, tokens, wall-clock, and how often it produced a usable result.

| Measure | Value |
| --- | --- |
| Samples | 100 |
| Completed (clean permutation produced) | 100/100 |
| True order recoverable (ordering metrics valid) | 99/100 |
| LLM lever-choice failures (fell back to defaults) | 0 |
| Total LLM calls | 400 |
| Avg LLM calls / sample | 4.00 |
| Total LLM tokens | 603129 |
| Avg LLM tokens / sample | 6031 |
| Avg wall-clock / sample (total) | 1m 20s |
| — Matched-budget control arm (no LLM) — |  |
| Control lever policy | random |
| Control LLM calls | 0 |
| Avg wall-clock / sample — agentic arm | 1m 0s |
| Avg wall-clock / sample — control arm | 19s |

## Per-Sample Results (showing first 15 and last 5 of 100; full table in `samples.jsonl`)
| Sample | Exact Match | Best Validity Score | Best Iteration | Fragments Placed | Junctions Pruned | Confirmed Adjacencies | Duration |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | no | 16.7996 | 3 | 23 | 4.3% | 17 | 43s |
| 2 | no | 17.2889 | 1 | 87 | 1.1% | 85 | 3m 11s |
| 3 | no | 15.9875 | 4 | 36 | 2.8% | 30 | 57s |
| 4 | no | 16.5510 | 2 | 31 | 3.2% | 31 | 50s |
| 5 | yes | 15.5481 | 2 | 8 | 12.5% | 6 | 43s |
| 6 | no | 20.2189 | 3 | 21 | 0.0% | 19 | 50s |
| 7 | no | 17.9481 | 4 | 36 | 2.8% | 29 | 59s |
| 8 | no | 15.5093 | 4 | 60 | 1.7% | 62 | 1m 45s |
| 9 | yes | 1.0000 | 1 | 11 | 9.1% | 10 | 46s |
| 10 | no | 15.3978 | 4 | 52 | 1.9% | 51 | 1m 31s |
| 11 | no | 15.8668 | 4 | 33 | 3.0% | 30 | 58s |
| 12 | no | 19.3007 | 2 | 13 | 0.0% | 10 | 37s |
| 13 | no | 17.1392 | 2 | 12 | 8.3% | 6 | 41s |
| 14 | no | 15.8927 | 4 | 83 | 1.2% | 83 | 3m 21s |
| 15 | yes | 1.0000 | 1 | 4 | 25.0% | 3 | 34s |
| 96 | yes | 19.0708 | 2 | 6 | 16.7% | 3 | 37s |
| 97 | no | 15.4589 | 5 | 23 | 4.3% | 19 | 42s |
| 98 | no | 14.8084 | 4 | 51 | 0.0% | 54 | 1m 17s |
| 99 | yes | 1.0000 | 1 | 6 | 0.0% | 5 | 38s |
| 100 | yes | 12.4560 | 2 | 12 | 8.3% | 10 | 32s |

## Quick Read
- Higher is better for every metric in the current set.
- Metrics: Exact Match (binary floor); Sequence Similarity (the one soft string metric); Adjacent Pair Accuracy (fraction of true fragment adjacencies preserved, the primary ordering metric); Longest Correct Run (longest contiguous correctly-ordered block, partial-assembly credit); Kendall Tau (global ordering correlation, 0 = random, 1 = perfect, -1 = reversed).
- Ordering metrics are NaN (skipped in the averages) for any sample whose fragments do not tile the target; the count of usable samples is in Cost, Efficiency & Completion.
- A positive delta means the reconstruction improved over the shuffled baseline.
- Each entry in samples.jsonl includes iteration_history with per-iteration lever_values and changed_levers for auditability.
- The validity score is the junction+overlap blended plausibility signal (lower = better); it measures plausibility, not exact-match correctness. Its trustworthiness is quantified in Validity Signal Concordance above.
- Junction-scorer ranking quality is measured separately and search-independently via `python -m evaluation.junction_ranking`.
- Use this report for side-by-side benchmarking; the raw per-sample data is in `samples.jsonl`.
