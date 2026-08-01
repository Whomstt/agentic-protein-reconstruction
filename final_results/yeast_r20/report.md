# Agentic Evaluation (Yeast, esm_small, r20)

## How to Read This Report
**Run type: Agentic (single_call).** Each metric compares:

- **Shuffled Baseline** — a random fragment ordering. The floor, not a method.
- **Deterministic (config defaults)** — the non-agentic baseline: iteration 1 run with the fixed `search.default_levers` and **no LLM call**. The agent refines from it.
- **Control (non-LLM)** — the matched-budget control arm: the SAME iteration budget, SAME fixed tool pipeline and SAME best-validity selection as the agentic arm, but the five levers are chosen by a non-LLM policy (random/grid) instead of the LLM, and it runs paired on the same protein with 0 LLM calls. **`Δ Agentic − Control` is the isolated value of the LLM's reasoning** — it separates "the agent reasons well" from "trying several candidates and keeping the best-validity one helps."
- **Agentic Best** — the agent's result: iterations 2+ are LLM lever choices, and the kept candidate is the best-validity one across all iterations (subject to `search.improvement_margin`). Since iteration 1 (the deterministic baseline) is in the candidate set, read the **true-metric** columns for the real "does the agent help?" answer.
- **Oracle (ceiling)** — for each metric, the best value achievable by selecting among the candidates the agent actually generated. Not a method (it peeks at the ground truth); the **Oracle − Agentic** gap is what the imperfect (~57–61%) validity concordance leaves on the table — reachable by better selection alone, no new candidate.

## Run Overview
- Samples evaluated: 100
- Avg junctions pruned: 5.1%
- Exact matches: 16/100
- Result folder: `140726_194805_agentic`
- Total run duration: 2h 18m 6s
- Avg duration per sample: 1m 22s

## Configuration
| Setting | Value |
| --- | --- |
| Run Method | agentic |
| Calling Mode | single_call |
| Device | cuda |
| Seed | 42 |
| Dataset | Yeast |
| Test Samples | 100 |
| Replica Count | 20 |
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
| Exact Match | 0.0100 | 0.1000 | 0.1500 | 0.1600 | 0.2300 | +0.0600 | +0.0100 | better |
| Sequence Similarity | 0.2178 | 0.3682 | 0.4005 | 0.4105 | 0.4671 | +0.0423 | +0.0101 | better |
| Adjacent Pair Accuracy | 0.0640 | 0.4992 | 0.6161 | 0.6417 | 0.6923 | +0.1425 | +0.0256 | better |
| Longest Correct Run | 0.1076 | 0.3003 | 0.3613 | 0.3772 | 0.4425 | +0.0769 | +0.0159 | better |
| Kendall Tau | 0.0444 | 0.3240 | 0.4285 | 0.4500 | 0.5468 | +0.1260 | +0.0215 | better |

![Metric comparison](metric_comparison.svg)

### Selection Ceiling (Oracle)
The Oracle column is the best true-metric value reachable by selecting among the candidates the agent already generated (it peeks at ground truth, so it is a ceiling, not a method). The gap below is quality the run left on the table purely to imperfect validity selection — reachable with better selection alone, no new candidate generated. A large gap says the bottleneck is the selection signal, not the search.

| Metric | Agentic Best (iteratively selected) | Oracle (best generated) | Gap (Oracle − Agentic) |
| --- | --- | --- | --- |
| Exact Match | 0.1600 | 0.2300 | +0.0700 |
| Sequence Similarity | 0.4105 | 0.4671 | +0.0566 |
| Adjacent Pair Accuracy | 0.6417 | 0.6923 | +0.0507 |
| Longest Correct Run | 0.3772 | 0.4425 | +0.0652 |
| Kendall Tau | 0.4500 | 0.5468 | +0.0968 |

## Agentic vs. Deterministic (paired, per sample, n=100)
Per-sample gain of the agentic result over the deterministic best-fixed baseline (Agentic − Deterministic on the same protein). Mean is the average improvement; std dev shows how consistently the agent helps vs. swings the other way on individual samples. For a significance claim on n samples, run a paired Wilcoxon signed-rank test on these per-sample gains.

| Metric | Mean Gain | Std Dev | Min Gain | Max Gain |
| --- | --- | --- | --- | --- |
| Exact Match | +0.0600 | 0.2764 | -1.0000 | +1.0000 |
| Sequence Similarity | +0.0423 | 0.1437 | -0.3873 | +0.6089 |
| Adjacent Pair Accuracy | +0.1425 | 0.1449 | -0.3333 | +0.7143 |
| Longest Correct Run | +0.0769 | 0.1430 | -0.3000 | +0.6250 |
| Kendall Tau | +0.1260 | 0.2595 | -0.4757 | +1.0476 |
| Best Validity Score (junction+overlap blend, lower=better) | -2.5647 | 1.8725 | -7.8293 | +0.0000 |

## Agentic vs. Control (paired, matched budget, n=100)
**The isolated value of the LLM's reasoning.** Per-sample gain of the agentic arm over the non-LLM control arm (Agentic − Control on the same protein), where both arms ran the same iteration budget, the same fixed tool pipeline and the same best-validity selection — only the lever *source* differed (LLM vs a random policy). A plain Agentic − Deterministic gain conflates 'the agent reasons well' with 'trying several candidates and keeping the best helps'; this comparison holds the budget and selection fixed, so a positive, consistent gain here is attributable to the LLM. Run a paired Wilcoxon signed-rank test on these per-sample gains for the significance claim.

| Metric | Mean Gain | Std Dev | Min Gain | Max Gain |
| --- | --- | --- | --- | --- |
| Exact Match | +0.0100 | 0.0995 | +0.0000 | +1.0000 |
| Sequence Similarity | +0.0101 | 0.0987 | -0.3941 | +0.5240 |
| Adjacent Pair Accuracy | +0.0256 | 0.0876 | -0.2609 | +0.3182 |
| Longest Correct Run | +0.0159 | 0.0963 | -0.3333 | +0.5000 |
| Kendall Tau | +0.0215 | 0.1573 | -0.4923 | +0.5621 |
| Best Validity Score (lower=better; negative = agentic more plausible) | -0.2811 | 0.7578 | -3.1050 | +2.2315 |

## Distribution Summary (n=100 samples)
The at-a-glance view for larger runs — read this before the per-sample table.

| Metric | Mean | Std Dev | Min | Median | Max |
| --- | --- | --- | --- | --- | --- |
| Exact Match | 0.1600 | 0.3666 | 0.0000 | 0.0000 | 1.0000 |
| Sequence Similarity | 0.4105 | 0.3603 | 0.0013 | 0.2489 | 1.0000 |
| Adjacent Pair Accuracy | 0.6417 | 0.1979 | 0.0000 | 0.6261 | 1.0000 |
| Longest Correct Run | 0.3772 | 0.3038 | 0.0400 | 0.2612 | 1.0000 |
| Kendall Tau | 0.4500 | 0.3408 | -0.1876 | 0.3966 | 1.0000 |
| Best Validity Score (junction+overlap blend, lower=better) | 15.9069 | 3.6242 | 1.0000 | 16.4859 | 26.2703 |

## Validity Signal Concordance
Whether the validity score used to *select* the winning candidate actually tracks true reconstruction quality, measured within each sample across the iterations it tried. 0.50 = no better than chance at picking the better of two candidates; higher is better. This is the trust check for the selection signal — if it is near 0.50, a better candidate the agent generated would not reliably be the one kept.

| Quality metric compared against | Concordance | Comparable pairs |
| --- | --- | --- |
| Kendall Tau | 0.627 | 119017 |
| Adjacent Pair Accuracy | 0.673 | 118254 |

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
| Total LLM tokens | 596101 |
| Avg LLM tokens / sample | 5961 |
| Avg wall-clock / sample (total) | 1m 22s |
| — Matched-budget control arm (no LLM) — |  |
| Control lever policy | random |
| Control LLM calls | 0 |
| Avg wall-clock / sample — agentic arm | 1m 3s |
| Avg wall-clock / sample — control arm | 19s |

## Per-Sample Results (showing first 15 and last 5 of 100; full table in `samples.jsonl`)
| Sample | Exact Match | Best Validity Score | Best Iteration | Fragments Placed | Junctions Pruned | Confirmed Adjacencies | Duration |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | no | 17.8959 | 5 | 55 | 1.8% | 45 | 2m 18s |
| 2 | no | 15.2837 | 4 | 91 | 1.1% | 60 | 3m 16s |
| 3 | no | 21.0041 | 2 | 41 | 2.4% | 31 | 55s |
| 4 | no | 21.3618 | 3 | 8 | 12.5% | 3 | 34s |
| 5 | no | 15.3705 | 5 | 48 | 2.1% | 32 | 1m 35s |
| 6 | no | 17.9481 | 5 | 15 | 6.7% | 10 | 38s |
| 7 | no | 18.8810 | 2 | 30 | 3.3% | 26 | 47s |
| 8 | no | 16.6401 | 5 | 50 | 2.0% | 34 | 1m 32s |
| 9 | no | 18.1130 | 4 | 34 | 2.9% | 28 | 51s |
| 10 | no | 15.1982 | 1 | 13 | 7.7% | 6 | 34s |
| 11 | no | 17.9753 | 3 | 57 | 1.8% | 45 | 1m 39s |
| 12 | yes | 16.4859 | 1 | 7 | 14.3% | 5 | 33s |
| 13 | no | 13.5122 | 2 | 33 | 3.0% | 19 | 51s |
| 14 | no | 18.1398 | 3 | 19 | 5.3% | 13 | 41s |
| 15 | no | 16.5547 | 2 | 9 | 11.1% | 4 | 33s |
| 96 | yes | 1.0000 | 1 | 6 | 16.7% | 5 | 33s |
| 97 | no | 15.9876 | 4 | 25 | 4.0% | 19 | 47s |
| 98 | no | 17.0228 | 2 | 69 | 1.4% | 49 | 1m 55s |
| 99 | no | 16.5831 | 3 | 40 | 2.5% | 32 | 1m 12s |
| 100 | no | 18.7714 | 2 | 39 | 0.0% | 34 | 1m 7s |

## Quick Read
- Higher is better for every metric in the current set.
- Metrics: Exact Match (binary floor); Sequence Similarity (the one soft string metric); Adjacent Pair Accuracy (fraction of true fragment adjacencies preserved, the primary ordering metric); Longest Correct Run (longest contiguous correctly-ordered block, partial-assembly credit); Kendall Tau (global ordering correlation, 0 = random, 1 = perfect, -1 = reversed).
- Ordering metrics are NaN (skipped in the averages) for any sample whose fragments do not tile the target; the count of usable samples is in Cost, Efficiency & Completion.
- A positive delta means the reconstruction improved over the shuffled baseline.
- Each entry in samples.jsonl includes iteration_history with per-iteration lever_values and changed_levers for auditability.
- The validity score is the junction+overlap blended plausibility signal (lower = better); it measures plausibility, not exact-match correctness. Its trustworthiness is quantified in Validity Signal Concordance above.
- Junction-scorer ranking quality is measured separately and search-independently via `python -m evaluation.junction_ranking`.
- Use this report for side-by-side benchmarking; the raw per-sample data is in `samples.jsonl`.
