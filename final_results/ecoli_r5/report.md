# Agentic Evaluation (E. coli, esm_small, r5)

## How to Read This Report
**Run type: Agentic (single_call).** Each metric compares:

- **Shuffled Baseline** — a random fragment ordering. The floor, not a method.
- **Deterministic (config defaults)** — the non-agentic baseline: iteration 1 run with the fixed `search.default_levers` and **no LLM call**. The agent refines from it.
- **Control (non-LLM)** — the matched-budget control arm: the SAME iteration budget, SAME fixed tool pipeline and SAME best-validity selection as the agentic arm, but the five levers are chosen by a non-LLM policy (random/grid) instead of the LLM, and it runs paired on the same protein with 0 LLM calls. **`Δ Agentic − Control` is the isolated value of the LLM's reasoning** — it separates "the agent reasons well" from "trying several candidates and keeping the best-validity one helps."
- **Agentic Best** — the agent's result: iterations 2+ are LLM lever choices, and the kept candidate is the best-validity one across all iterations (subject to `search.improvement_margin`). Since iteration 1 (the deterministic baseline) is in the candidate set, read the **true-metric** columns for the real "does the agent help?" answer.
- **Oracle (ceiling)** — for each metric, the best value achievable by selecting among the candidates the agent actually generated. Not a method (it peeks at the ground truth); the **Oracle − Agentic** gap is what the imperfect (~57–61%) validity concordance leaves on the table — reachable by better selection alone, no new candidate.

## Run Overview
- Samples evaluated: 100
- Avg junctions pruned: 6.3%
- Exact matches: 9/100
- Result folder: `140726_144552_agentic`
- Total run duration: 1h 13m 10s
- Avg duration per sample: 43s

## Configuration
| Setting | Value |
| --- | --- |
| Run Method | agentic |
| Calling Mode | single_call |
| Device | cuda |
| Seed | 42 |
| Dataset | E. coli |
| Test Samples | 100 |
| Replica Count | 5 |
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
| Exact Match | 0.0200 | 0.0500 | 0.0900 | 0.0900 | 0.1200 | +0.0400 | +0.0000 | better |
| Sequence Similarity | 0.2995 | 0.3908 | 0.4588 | 0.4531 | 0.5143 | +0.0622 | -0.0058 | better |
| Adjacent Pair Accuracy | 0.1012 | 0.2937 | 0.4357 | 0.4328 | 0.4926 | +0.1392 | -0.0029 | better |
| Longest Correct Run | 0.1591 | 0.2294 | 0.2967 | 0.2889 | 0.3424 | +0.0595 | -0.0078 | better |
| Kendall Tau | -0.0121 | 0.2483 | 0.4013 | 0.4043 | 0.4872 | +0.1560 | +0.0030 | better |

![Metric comparison](metric_comparison.svg)

### Selection Ceiling (Oracle)
The Oracle column is the best true-metric value reachable by selecting among the candidates the agent already generated (it peeks at ground truth, so it is a ceiling, not a method). The gap below is quality the run left on the table purely to imperfect validity selection — reachable with better selection alone, no new candidate generated. A large gap says the bottleneck is the selection signal, not the search.

| Metric | Agentic Best (iteratively selected) | Oracle (best generated) | Gap (Oracle − Agentic) |
| --- | --- | --- | --- |
| Exact Match | 0.0900 | 0.1200 | +0.0300 |
| Sequence Similarity | 0.4531 | 0.5143 | +0.0613 |
| Adjacent Pair Accuracy | 0.4328 | 0.4926 | +0.0598 |
| Longest Correct Run | 0.2889 | 0.3424 | +0.0536 |
| Kendall Tau | 0.4043 | 0.4872 | +0.0829 |

## Agentic vs. Deterministic (paired, per sample, n=100)
Per-sample gain of the agentic result over the deterministic best-fixed baseline (Agentic − Deterministic on the same protein). Mean is the average improvement; std dev shows how consistently the agent helps vs. swings the other way on individual samples. For a significance claim on n samples, run a paired Wilcoxon signed-rank test on these per-sample gains.

| Metric | Mean Gain | Std Dev | Min Gain | Max Gain |
| --- | --- | --- | --- | --- |
| Exact Match | +0.0400 | 0.1960 | +0.0000 | +1.0000 |
| Sequence Similarity | +0.0622 | 0.1723 | -0.4578 | +0.6244 |
| Adjacent Pair Accuracy | +0.1392 | 0.1522 | -0.2857 | +0.5714 |
| Longest Correct Run | +0.0595 | 0.1332 | -0.2500 | +0.6000 |
| Kendall Tau | +0.1560 | 0.2545 | -0.2909 | +1.0000 |
| Best Validity Score (junction+overlap blend, lower=better) | -4.8612 | 3.5924 | -15.8110 | +0.0000 |

## Agentic vs. Control (paired, matched budget, n=100)
**The isolated value of the LLM's reasoning.** Per-sample gain of the agentic arm over the non-LLM control arm (Agentic − Control on the same protein), where both arms ran the same iteration budget, the same fixed tool pipeline and the same best-validity selection — only the lever *source* differed (LLM vs a random policy). A plain Agentic − Deterministic gain conflates 'the agent reasons well' with 'trying several candidates and keeping the best helps'; this comparison holds the budget and selection fixed, so a positive, consistent gain here is attributable to the LLM. Run a paired Wilcoxon signed-rank test on these per-sample gains for the significance claim.

| Metric | Mean Gain | Std Dev | Min Gain | Max Gain |
| --- | --- | --- | --- | --- |
| Exact Match | +0.0000 | 0.0000 | +0.0000 | +0.0000 |
| Sequence Similarity | -0.0058 | 0.1104 | -0.3790 | +0.5408 |
| Adjacent Pair Accuracy | -0.0029 | 0.0760 | -0.3333 | +0.1538 |
| Longest Correct Run | -0.0078 | 0.0383 | -0.2000 | +0.0870 |
| Kendall Tau | +0.0030 | 0.1370 | -0.4000 | +0.6970 |
| Best Validity Score (lower=better; negative = agentic more plausible) | -0.3809 | 1.0765 | -6.1047 | +1.0579 |

## Distribution Summary (n=100 samples)
The at-a-glance view for larger runs — read this before the per-sample table.

| Metric | Mean | Std Dev | Min | Median | Max |
| --- | --- | --- | --- | --- | --- |
| Exact Match | 0.0900 | 0.2862 | 0.0000 | 0.0000 | 1.0000 |
| Sequence Similarity | 0.4531 | 0.3532 | 0.0018 | 0.3803 | 1.0000 |
| Adjacent Pair Accuracy | 0.4328 | 0.2267 | 0.1250 | 0.3798 | 1.0000 |
| Longest Correct Run | 0.2889 | 0.2501 | 0.0488 | 0.2053 | 1.0000 |
| Kendall Tau | 0.4043 | 0.3204 | -0.3091 | 0.3959 | 1.0000 |
| Best Validity Score (junction+overlap blend, lower=better) | 14.7345 | 2.9335 | 1.0000 | 14.9886 | 21.5421 |

## Validity Signal Concordance
Whether the validity score used to *select* the winning candidate actually tracks true reconstruction quality, measured within each sample across the iterations it tried. 0.50 = no better than chance at picking the better of two candidates; higher is better. This is the trust check for the selection signal — if it is near 0.50, a better candidate the agent generated would not reliably be the one kept.

| Quality metric compared against | Concordance | Comparable pairs |
| --- | --- | --- |
| Kendall Tau | 0.550 | 123604 |
| Adjacent Pair Accuracy | 0.596 | 122315 |

## Cost, Efficiency & Completion
The non-quality axis the agentic approach must also justify itself on: LLM calls, tokens, wall-clock, and how often it produced a usable result.

| Measure | Value |
| --- | --- |
| Samples | 100 |
| Completed (clean permutation produced) | 100/100 |
| True order recoverable (ordering metrics valid) | 100/100 |
| LLM lever-choice failures (fell back to defaults) | 0 |
| Total LLM calls | 400 |
| Avg LLM calls / sample | 4.00 |
| Total LLM tokens | 595999 |
| Avg LLM tokens / sample | 5960 |
| Avg wall-clock / sample (total) | 43s |
| — Matched-budget control arm (no LLM) — |  |
| Control lever policy | random |
| Control LLM calls | 0 |
| Avg wall-clock / sample — agentic arm | 37s |
| Avg wall-clock / sample — control arm | 5s |

## Per-Sample Results (showing first 15 and last 5 of 100; full table in `samples.jsonl`)
| Sample | Exact Match | Best Validity Score | Best Iteration | Fragments Placed | Junctions Pruned | Confirmed Adjacencies | Duration |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | no | 12.0421 | 3 | 13 | 7.7% | 2 | 41s |
| 2 | yes | 17.9656 | 2 | 5 | 20.0% | 0 | 36s |
| 3 | no | 13.1257 | 5 | 35 | 2.9% | 9 | 55s |
| 4 | no | 16.5708 | 5 | 14 | 0.0% | 2 | 36s |
| 5 | no | 18.0805 | 2 | 10 | 10.0% | 2 | 32s |
| 6 | yes | 18.1196 | 2 | 6 | 16.7% | 2 | 29s |
| 7 | no | 15.8761 | 2 | 33 | 3.0% | 4 | 59s |
| 8 | no | 15.8844 | 2 | 9 | 11.1% | 1 | 35s |
| 9 | no | 17.1361 | 4 | 11 | 9.1% | 2 | 35s |
| 10 | no | 20.0158 | 2 | 5 | 20.0% | 0 | 31s |
| 11 | no | 15.3048 | 3 | 16 | 0.0% | 5 | 35s |
| 12 | yes | 16.7541 | 2 | 6 | 16.7% | 4 | 33s |
| 13 | no | 12.0971 | 3 | 11 | 0.0% | 2 | 36s |
| 14 | no | 16.1744 | 4 | 27 | 3.7% | 7 | 1m 4s |
| 15 | no | 14.0575 | 4 | 14 | 7.1% | 2 | 34s |
| 96 | no | 16.3818 | 2 | 12 | 8.3% | 3 | 37s |
| 97 | no | 11.0782 | 3 | 15 | 6.7% | 4 | 35s |
| 98 | no | 13.7162 | 4 | 20 | 5.0% | 4 | 43s |
| 99 | no | 12.2298 | 3 | 13 | 7.7% | 6 | 36s |
| 100 | no | 13.8180 | 3 | 12 | 8.3% | 2 | 37s |

## Quick Read
- Higher is better for every metric in the current set.
- Metrics: Exact Match (binary floor); Sequence Similarity (the one soft string metric); Adjacent Pair Accuracy (fraction of true fragment adjacencies preserved, the primary ordering metric); Longest Correct Run (longest contiguous correctly-ordered block, partial-assembly credit); Kendall Tau (global ordering correlation, 0 = random, 1 = perfect, -1 = reversed).
- Ordering metrics are NaN (skipped in the averages) for any sample whose fragments do not tile the target; the count of usable samples is in Cost, Efficiency & Completion.
- A positive delta means the reconstruction improved over the shuffled baseline.
- Each entry in samples.jsonl includes iteration_history with per-iteration lever_values and changed_levers for auditability.
- The validity score is the junction+overlap blended plausibility signal (lower = better); it measures plausibility, not exact-match correctness. Its trustworthiness is quantified in Validity Signal Concordance above.
- Junction-scorer ranking quality is measured separately and search-independently via `python -m evaluation.junction_ranking`.
- Use this report for side-by-side benchmarking; the raw per-sample data is in `samples.jsonl`.
