# Agentic Evaluation (Yeast, esm_small, r5) — Statistical Report

_Generated 2026-08-01 18:02 by `python -m evaluation.rebuild`. Every number is computed from `samples.jsonl` (n=100); nothing is transcribed by hand. Bootstraps use a fixed seed, so a rebuild reproduces this file exactly._

| Setting | Value |
| --- | --- |
| Organism | Yeast |
| Digestion replicas | 5 |
| Samples | 100 |
| Missed cleavage ratio | 0.3 |
| Method / calling mode | agentic / single_call |
| Iteration 1 deterministic | True |
| Max iterations | 5 |
| Improvement margin | n/a |
| MLM | facebook/esm2_t6_8M_UR50D |
| LLM | gpt-5-mini |
| Bootstrap | 10000 resamples, seed 20260726 |

**Disclosures.** There is no train/test split in this project — constants such as `improvement_margin`, `validity_junction_window` and `validity_confirmed_penalty` were chosen on samples drawn from the same undivided pool this evaluation draws from, so they are disclosed sensitivity choices, not validated constants. Iteration 1 being deterministic means the agentic arm can never score worse than the Fixed Settings arm on validity, which is why section D's `LLM-Guided − Random Search` comparison is the defensible reasoning claim.

## A. Overall Performance

All values are means over n=100 proteins with 95% confidence intervals. Exact Match uses a **Wilson score interval** (it is a count of successes out of n, not a continuous mean); the other four use a **BCa bootstrap** (10000 resamples, fixed seed 20260726, so a rebuild reproduces these intervals exactly).

| Metric | Random Order | Fixed Settings | Random Search (no LLM) | LLM-Guided | Best Candidate (ceiling) |
| --- | --- | --- | --- | --- | --- |
| Exact Match | 0.0200 [0.0055, 0.0700] | 0.0500 [0.0215, 0.1118] | 0.1000 [0.0552, 0.1744] | 0.1100 [0.0625, 0.1863] | 0.1400 [0.0853, 0.2214] |
| Sequence Similarity | 0.2737 [0.2197, 0.3343] | 0.3437 [0.2811, 0.4115] | 0.3550 [0.2884, 0.4288] | 0.3651 [0.2982, 0.4395] | 0.4248 [0.3540, 0.5001] |
| Edit Similarity | n/a | 0.4138 [0.3752, 0.4606] | 0.4464 [0.4005, 0.4988] | 0.4567 [0.4099, 0.5099] | 0.5394 [0.4934, 0.5904] |
| Adjacent Pair Accuracy | 0.0712 [0.0500, 0.1138] | 0.2700 [0.2335, 0.3182] | 0.3840 [0.3391, 0.4363] | 0.3972 [0.3523, 0.4498] | 0.4493 [0.4030, 0.5026] |
| Longest Correct Run | 0.1198 [0.0965, 0.1592] | 0.2047 [0.1660, 0.2554] | 0.2522 [0.2055, 0.3120] | 0.2656 [0.2168, 0.3277] | 0.3070 [0.2521, 0.3709] |
| Kendall Tau | 0.0090 [-0.0308, 0.0580] | 0.1936 [0.1455, 0.2514] | 0.3100 [0.2505, 0.3758] | 0.3243 [0.2609, 0.3933] | 0.4363 [0.3771, 0.5000] |

## B. Method Ladder

| Metric | Random Order | Fixed Settings | Random Search (no LLM) | LLM-Guided | Best Candidate (ceiling) |
| --- | --- | --- | --- | --- | --- |
| Exact Match | 0.0200 | 0.0500 (+0.0300) | 0.1000 (+0.0500) | 0.1100 (+0.0100) | 0.1400 (+0.0300) |
| Sequence Similarity | 0.2737 | 0.3437 (+0.0700) | 0.3550 (+0.0113) | 0.3651 (+0.0101) | 0.4248 (+0.0597) |
| Edit Similarity | n/a | 0.4138 (n/a) | 0.4464 (+0.0326) | 0.4567 (+0.0103) | 0.5394 (+0.0827) |
| Adjacent Pair Accuracy | 0.0712 | 0.2700 (+0.1988) | 0.3840 (+0.1140) | 0.3972 (+0.0132) | 0.4493 (+0.0521) |
| Longest Correct Run | 0.1198 | 0.2047 (+0.0849) | 0.2522 (+0.0475) | 0.2656 (+0.0134) | 0.3070 (+0.0415) |
| Kendall Tau | 0.0090 | 0.1936 (+0.1846) | 0.3100 (+0.1164) | 0.3243 (+0.0143) | 0.4363 (+0.1120) |

### Reading the Sequence Similarity floor

**A random shuffle of the fragments already scores 0.2737 on Sequence Similarity.** Every candidate ordering is a permutation of the *same* fragment multiset, so the string composition is identical across all arms and only the order varies. `difflib.SequenceMatcher` credits matching blocks wherever they occur, so a large fraction of that ratio is bought by composition alone and is available to a method that has learned nothing.

For contrast, on the same shuffled orderings the ordering-sensitive metrics sit at their true floor: Exact Match 0.0200, Adjacent Pair Accuracy 0.0712, Kendall Tau 0.0090. **Read Sequence Similarity only as a delta against the shuffled floor, never as an absolute.** Adjacent Pair Accuracy is the primary ordering metric.

## C. Replica Scaling

This run sits at **5 digestion replicas**. Replica count is what determines how many adjacencies the overlap graph can confirm outright, which is near-ground-truth structural information the search gets for free.

| Quantity | Value |
| --- | --- |
| Digestion replicas | 5 |
| Mean confirmed adjacencies per protein | 9.97 |
| Mean true joins covered by the overlap graph | 0.3129 |
| Mean junctions pruned by trypsin filter (%) | 6.26 |

_The scaling curve across replica counts is in the cross-run report_ (`cross_run_report.md`), _which needs more than one run to draw._

## D. Isolating the LLM's Contribution

The LLM-Guided and Random Search arms run on the **same proteins** with the same iteration budget, the same tool pipeline and the same best-validity selection; only the source of the five lever values differs (LLM vs. a non-LLM policy). They are therefore **paired**, and the comparison uses paired tests rather than asking whether the two arms' confidence intervals overlap — an overlap test on paired data is both wrong and badly underpowered.

Exact Match uses an **exact McNemar test** on the discordant pairs; the four continuous metrics use a **Wilcoxon signed-rank test**. Because five metrics are tested on one hypothesis, p-values are corrected with **Holm** across the family.

### LLM-Guided − Random Search

| Metric | Mean Δ | 95% CI | Test | Discordant / non-zero pairs | p | p (Holm) | Significant |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Exact Match | +0.0100 | [+0.0000, +0.0300] | McNemar (exact) | 1 (A only 1, B only 0) | 1.000 | 1.000 | no |
| Sequence Similarity | +0.0101 | [-0.0032, +0.0263] | Wilcoxon | 52 non-zero (27+ / 25-) | 0.316 | 1.000 | no |
| Edit Similarity | +0.0103 | [-0.0081, +0.0350] | Wilcoxon | 73 non-zero (37+ / 36-) | 0.773 | 1.000 | no |
| Adjacent Pair Accuracy | +0.0132 | [-0.0002, +0.0304] | Wilcoxon | 60 non-zero (33+ / 27-) | 0.273 | 1.000 | no |
| Longest Correct Run | +0.0134 | [+0.0007, +0.0412] | Wilcoxon | 34 non-zero (21+ / 13-) | 0.620 | 1.000 | no |
| Kendall Tau | +0.0143 | [-0.0123, +0.0467] | Wilcoxon | 73 non-zero (40+ / 33-) | 0.773 | 1.000 | no |

**No metric survives Holm correction.** The observed differences are consistent with what re-running the same non-LLM policy on these n=100 proteins could produce by chance; this run does not demonstrate a reasoning advantage on this comparison.

### LLM-Guided − Fixed Settings

| Metric | Mean Δ | 95% CI | Test | Discordant / non-zero pairs | p | p (Holm) | Significant |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Exact Match | +0.0600 | [+0.0100, +0.1100] | McNemar (exact) | 8 (A only 7, B only 1) | 0.070 | 0.149 | no |
| Sequence Similarity | +0.0214 | [-0.0033, +0.0465] | Wilcoxon | 73 non-zero (38+ / 35-) | 0.174 | 0.174 | no |
| Edit Similarity | +0.0429 | [+0.0109, +0.0790] | Wilcoxon | 90 non-zero (53+ / 37-) | 0.050 | 0.149 | no |
| Adjacent Pair Accuracy | +0.1272 | [+0.0969, +0.1643] | Wilcoxon | 81 non-zero (76+ / 5-) | <0.001 | <0.001 | yes |
| Longest Correct Run | +0.0609 | [+0.0329, +0.0973] | Wilcoxon | 69 non-zero (60+ / 9-) | <0.001 | <0.001 | yes |
| Kendall Tau | +0.1308 | [+0.0777, +0.1920] | Wilcoxon | 87 non-zero (55+ / 32-) | <0.001 | <0.001 | yes |

Significant after Holm correction: **Adjacent Pair Accuracy, Longest Correct Run, Kendall Tau**.

## E. Where the Bottleneck Is

### Junction scorer ranking (search-independent)

n/a - requires field: `junction_ranking` (per-sample top-1/top-3/MRR). The dense junction score matrix is not stored in samples.jsonl, and recomputing it needs pLM inference, which this offline rebuild does not do. Runs from the instrumentation change onward record it at zero extra model cost; for older runs, `python -m evaluation.junction_ranking` measures it separately.

### Selection signal trust

The run keeps whichever candidate scores best on the validity signal, so the signal's ability to rank candidates bounds what the search can deliver. **0.50 is a coin flip.**

| Measurement | Value |
| --- | --- |
| Samples with comparable candidate pairs | 96 |
| Comparable candidate pairs | 709 |
| Mean within-sample concordance | 0.7635 |
| Samples where concordance > 0.50 | 80 |
| Validity junction window | n/a |
| Validity confirmed penalty | n/a |

### Selection ceiling (Best Candidate)

| Metric | LLM-Guided | Best Candidate | Gap | Samples with a gap |
| --- | --- | --- | --- | --- |
| Exact Match | 0.1100 | 0.1400 | +0.0300 | 3/100 |
| Sequence Similarity | 0.3651 | 0.4248 | +0.0597 | 61/100 |
| Edit Similarity | 0.4567 | 0.5394 | +0.0827 | 68/100 |
| Adjacent Pair Accuracy | 0.3972 | 0.4493 | +0.0521 | 41/100 |
| Longest Correct Run | 0.2656 | 0.3070 | +0.0415 | 28/100 |
| Kendall Tau | 0.3243 | 0.4363 | +0.1120 | 69/100 |

On Adjacent Pair Accuracy the run leaves **0.0521** on the table in candidates it had already generated but did not select (41/100 samples). That is the size of the prize for a better selection signal alone.

### Trypsin filter recall

The filter pruned **6.26%** of candidate junctions on average. Whether any pruned junction was a *true* one is n/a - requires field: `trypsin_recall` (which junctions the trypsin filter pruned). Older runs stored only the pruned COUNT (`num_pruned`), which cannot tell us whether a pruned junction was a true one. Recorded from the instrumentation change onward.

## F. Difficulty Stratification and Error Modes

### Adjacent Pair Accuracy by fragment count

Difficulty scales with how many pieces the protein was cut into: the number of possible orderings grows factorially, while the pLM's evidence per junction does not improve. Lift over the Random Order floor is the honest read of whether the method is doing anything at each difficulty.

| Fragments | n | Random Order | LLM-Guided | Lift (paired) |
| --- | --- | --- | --- | --- |
| 2-4 | 4 | 0.5000 | 1.0000 | +0.5000 |
| 5-9 | 18 | 0.0765 | 0.5682 | +0.4917 |
| 10-19 | 21 | 0.0920 | 0.4314 | +0.3393 |
| 20-49 | 37 | 0.0365 | 0.3100 | +0.2735 |
| 50+ | 20 | 0.0231 | 0.2481 | +0.2250 |

### Every metric by fragment count

| Metric | 2-4 | 5-9 | 10-19 | 20-49 | 50+ |
| --- | --- | --- | --- | --- | --- |
| Exact Match | 1.0000 | 0.3333 | 0.0476 | 0.0000 | 0.0000 |
| Sequence Similarity | 1.0000 | 0.8074 | 0.6064 | 0.1224 | 0.0356 |
| Edit Similarity | 1.0000 | 0.7193 | 0.5473 | 0.3246 | 0.2611 |
| Adjacent Pair Accuracy | 1.0000 | 0.5682 | 0.4314 | 0.3100 | 0.2481 |
| Longest Correct Run | 1.0000 | 0.5765 | 0.3029 | 0.1231 | 0.0633 |
| Kendall Tau | 1.0000 | 0.6453 | 0.4030 | 0.1866 | 0.0727 |

### N-terminal start

| Measurement | Value |
| --- | --- |
| P(correct N-terminal start) | 0.8100 |
| ...on shuffled orderings | 0.1000 |
| Exact Match \| correct start | 0.1358 (11/81) |
| Exact Match \| wrong start | 0.0000 (0/19) |

Exact reconstruction is effectively conditional on getting the first fragment right — an ordering that starts wrong has already displaced every fragment after it.

### Breakpoints

| Measurement | Value |
| --- | --- |
| Samples | 100 |
| Mean breakpoints per protein | 21.99 |
| Median | 17.00 |
| Min / Max | 0 / 91 |
| Mean breakpoints per join | 0.6028 |
| Proteins assembled with 0 breakpoints | 11 |

### Error taxonomy

Failures are classified from the stored metric values (which were computed with the correct fragment-string semantics), checked in a fixed order so each sample lands in exactly one class. The cut points are disclosed in the table note and affect labelling only — no headline number depends on them.

| Failure mode | Samples | Share |
| --- | --- | --- |
| Exact reconstruction | 11 | 11.0% |
| Local transposition | 1 | 1.0% |
| Wrong start (structured, misanchored) | 19 | 19.0% |
| Partial assembly (correct start) | 69 | 69.0% |

## G. Cost

| Measurement | Value |
| --- | --- |
| LLM model | gpt-5-mini |
| Total LLM calls | 400 |
| Total LLM tokens | 589057 |
| LLM calls per sample | 4.00 |
| LLM tokens per sample | 5890.6 |
| Lever-choice failures | 0 |
| Wall clock per sample (total) | 70.9 s |
| Wall clock per sample (agentic arm) | 54.8 s |
| Wall clock per sample (control arm) | 15.9 s |
| LLM-Guided / Random Search time ratio | 3.44 |
| Completed samples | 100/100 |
| True order recovered | 100/100 |

The agentic arm costs **3.4x** the control arm's wall clock and **4.0 LLM calls per sample**, against which section D's paired tests are the return.

## Provenance

- Source: `samples.jsonl` in this folder — per-sample stored data only.
- No GPU, no model loading and no network access were used to build this report.
- Regenerate with `python -m evaluation.rebuild --run 140726_215821_agentic`.
- `report.md` (the run's original report) is left untouched; this file is additive.
