# Agentic Evaluation (E. coli, esm_small, r5) — Statistical Report

_Generated 2026-08-01 18:02 by `python -m evaluation.rebuild`. Every number is computed from `samples.jsonl` (n=100); nothing is transcribed by hand. Bootstraps use a fixed seed, so a rebuild reproduces this file exactly._

| Setting | Value |
| --- | --- |
| Organism | E. coli |
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
| Exact Match | 0.0200 [0.0055, 0.0700] | 0.0500 [0.0215, 0.1118] | 0.0900 [0.0481, 0.1623] | 0.0900 [0.0481, 0.1623] | 0.1200 [0.0700, 0.1981] |
| Sequence Similarity | 0.2995 [0.2481, 0.3580] | 0.3908 [0.3304, 0.4548] | 0.4588 [0.3907, 0.5272] | 0.4531 [0.3845, 0.5236] | 0.5143 [0.4456, 0.5815] |
| Edit Similarity | n/a | 0.4494 [0.4112, 0.4942] | 0.5483 [0.5042, 0.5969] | 0.5375 [0.4926, 0.5883] | 0.6097 [0.5653, 0.6546] |
| Adjacent Pair Accuracy | 0.1012 [0.0734, 0.1446] | 0.2937 [0.2538, 0.3420] | 0.4357 [0.3943, 0.4845] | 0.4328 [0.3916, 0.4804] | 0.4926 [0.4494, 0.5400] |
| Longest Correct Run | 0.1591 [0.1313, 0.2016] | 0.2294 [0.1949, 0.2785] | 0.2967 [0.2538, 0.3524] | 0.2889 [0.2460, 0.3448] | 0.3424 [0.2931, 0.4010] |
| Kendall Tau | -0.0121 [-0.0686, 0.0453] | 0.2483 [0.1941, 0.3091] | 0.4013 [0.3402, 0.4645] | 0.4043 [0.3417, 0.4670] | 0.4872 [0.4308, 0.5443] |

## B. Method Ladder

| Metric | Random Order | Fixed Settings | Random Search (no LLM) | LLM-Guided | Best Candidate (ceiling) |
| --- | --- | --- | --- | --- | --- |
| Exact Match | 0.0200 | 0.0500 (+0.0300) | 0.0900 (+0.0400) | 0.0900 (+0.0000) | 0.1200 (+0.0300) |
| Sequence Similarity | 0.2995 | 0.3908 (+0.0913) | 0.4588 (+0.0680) | 0.4531 (-0.0058) | 0.5143 (+0.0613) |
| Edit Similarity | n/a | 0.4494 (n/a) | 0.5483 (+0.0989) | 0.5375 (-0.0108) | 0.6097 (+0.0721) |
| Adjacent Pair Accuracy | 0.1012 | 0.2937 (+0.1924) | 0.4357 (+0.1421) | 0.4328 (-0.0029) | 0.4926 (+0.0598) |
| Longest Correct Run | 0.1591 | 0.2294 (+0.0703) | 0.2967 (+0.0673) | 0.2889 (-0.0078) | 0.3424 (+0.0536) |
| Kendall Tau | -0.0121 | 0.2483 (+0.2604) | 0.4013 (+0.1530) | 0.4043 (+0.0030) | 0.4872 (+0.0829) |

### Reading the Sequence Similarity floor

**A random shuffle of the fragments already scores 0.2995 on Sequence Similarity.** Every candidate ordering is a permutation of the *same* fragment multiset, so the string composition is identical across all arms and only the order varies. `difflib.SequenceMatcher` credits matching blocks wherever they occur, so a large fraction of that ratio is bought by composition alone and is available to a method that has learned nothing.

For contrast, on the same shuffled orderings the ordering-sensitive metrics sit at their true floor: Exact Match 0.0200, Adjacent Pair Accuracy 0.1012, Kendall Tau -0.0121. **Read Sequence Similarity only as a delta against the shuffled floor, never as an absolute.** Adjacent Pair Accuracy is the primary ordering metric.

## C. Replica Scaling

This run sits at **5 digestion replicas**. Replica count is what determines how many adjacencies the overlap graph can confirm outright, which is near-ground-truth structural information the search gets for free.

| Quantity | Value |
| --- | --- |
| Digestion replicas | 5 |
| Mean confirmed adjacencies per protein | 4.76 |
| Mean true joins covered by the overlap graph | 0.2966 |
| Mean junctions pruned by trypsin filter (%) | 6.29 |

_The scaling curve across replica counts is in the cross-run report_ (`cross_run_report.md`), _which needs more than one run to draw._

## D. Isolating the LLM's Contribution

The LLM-Guided and Random Search arms run on the **same proteins** with the same iteration budget, the same tool pipeline and the same best-validity selection; only the source of the five lever values differs (LLM vs. a non-LLM policy). They are therefore **paired**, and the comparison uses paired tests rather than asking whether the two arms' confidence intervals overlap — an overlap test on paired data is both wrong and badly underpowered.

Exact Match uses an **exact McNemar test** on the discordant pairs; the four continuous metrics use a **Wilcoxon signed-rank test**. Because five metrics are tested on one hypothesis, p-values are corrected with **Holm** across the family.

### LLM-Guided − Random Search

| Metric | Mean Δ | 95% CI | Test | Discordant / non-zero pairs | p | p (Holm) | Significant |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Exact Match | +0.0000 | [+0.0000, +0.0000] | McNemar (exact) | 0 (A only 0, B only 0) | 1.000 | 1.000 | no |
| Sequence Similarity | -0.0058 | [-0.0274, +0.0162] | Wilcoxon | 44 non-zero (20+ / 24-) | 0.623 | 1.000 | no |
| Edit Similarity | -0.0108 | [-0.0273, +0.0055] | Wilcoxon | 64 non-zero (26+ / 38-) | 0.130 | 0.649 | no |
| Adjacent Pair Accuracy | -0.0029 | [-0.0195, +0.0108] | Wilcoxon | 49 non-zero (25+ / 24-) | 0.909 | 1.000 | no |
| Longest Correct Run | -0.0078 | [-0.0164, -0.0011] | Wilcoxon | 23 non-zero (8+ / 15-) | 0.053 | 0.320 | no |
| Kendall Tau | +0.0030 | [-0.0220, +0.0316] | Wilcoxon | 64 non-zero (35+ / 29-) | 0.878 | 1.000 | no |

**No metric survives Holm correction.** The observed differences are consistent with what re-running the same non-LLM policy on these n=100 proteins could produce by chance; this run does not demonstrate a reasoning advantage on this comparison.

### LLM-Guided − Fixed Settings

| Metric | Mean Δ | 95% CI | Test | Discordant / non-zero pairs | p | p (Holm) | Significant |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Exact Match | +0.0400 | [+0.0100, +0.0800] | McNemar (exact) | 4 (A only 4, B only 0) | 0.125 | 0.125 | no |
| Sequence Similarity | +0.0622 | [+0.0299, +0.0987] | Wilcoxon | 82 non-zero (48+ / 34-) | 0.002 | 0.003 | yes |
| Edit Similarity | +0.0881 | [+0.0536, +0.1261] | Wilcoxon | 92 non-zero (62+ / 30-) | <0.001 | <0.001 | yes |
| Adjacent Pair Accuracy | +0.1392 | [+0.1091, +0.1695] | Wilcoxon | 78 non-zero (72+ / 6-) | <0.001 | <0.001 | yes |
| Longest Correct Run | +0.0595 | [+0.0368, +0.0899] | Wilcoxon | 56 non-zero (47+ / 9-) | <0.001 | <0.001 | yes |
| Kendall Tau | +0.1560 | [+0.1080, +0.2085] | Wilcoxon | 93 non-zero (67+ / 26-) | <0.001 | <0.001 | yes |

Significant after Holm correction: **Sequence Similarity, Edit Similarity, Adjacent Pair Accuracy, Longest Correct Run, Kendall Tau**.

## E. Where the Bottleneck Is

### Junction scorer ranking (search-independent)

n/a - requires field: `junction_ranking` (per-sample top-1/top-3/MRR). The dense junction score matrix is not stored in samples.jsonl, and recomputing it needs pLM inference, which this offline rebuild does not do. Runs from the instrumentation change onward record it at zero extra model cost; for older runs, `python -m evaluation.junction_ranking` measures it separately.

### Selection signal trust

The run keeps whichever candidate scores best on the validity signal, so the signal's ability to rank candidates bounds what the search can deliver. **0.50 is a coin flip.**

| Measurement | Value |
| --- | --- |
| Samples with comparable candidate pairs | 92 |
| Comparable candidate pairs | 667 |
| Mean within-sample concordance | 0.7521 |
| Samples where concordance > 0.50 | 81 |
| Validity junction window | n/a |
| Validity confirmed penalty | n/a |

### Selection ceiling (Best Candidate)

| Metric | LLM-Guided | Best Candidate | Gap | Samples with a gap |
| --- | --- | --- | --- | --- |
| Exact Match | 0.0900 | 0.1200 | +0.0300 | 3/100 |
| Sequence Similarity | 0.4531 | 0.5143 | +0.0613 | 56/100 |
| Edit Similarity | 0.5375 | 0.6097 | +0.0721 | 64/100 |
| Adjacent Pair Accuracy | 0.4328 | 0.4926 | +0.0598 | 47/100 |
| Longest Correct Run | 0.2889 | 0.3424 | +0.0536 | 34/100 |
| Kendall Tau | 0.4043 | 0.4872 | +0.0829 | 60/100 |

On Adjacent Pair Accuracy the run leaves **0.0598** on the table in candidates it had already generated but did not select (47/100 samples). That is the size of the prize for a better selection signal alone.

### Trypsin filter recall

The filter pruned **6.29%** of candidate junctions on average. Whether any pruned junction was a *true* one is n/a - requires field: `trypsin_recall` (which junctions the trypsin filter pruned). Older runs stored only the pruned COUNT (`num_pruned`), which cannot tell us whether a pruned junction was a true one. Recorded from the instrumentation change onward.

## F. Difficulty Stratification and Error Modes

### Adjacent Pair Accuracy by fragment count

Difficulty scales with how many pieces the protein was cut into: the number of possible orderings grows factorially, while the pLM's evidence per junction does not improve. Lift over the Random Order floor is the honest read of whether the method is doing anything at each difficulty.

| Fragments | n | Random Order | LLM-Guided | Lift (paired) |
| --- | --- | --- | --- | --- |
| 2-4 | 5 | 0.5000 | 1.0000 | +0.5000 |
| 5-9 | 14 | 0.2212 | 0.5857 | +0.3645 |
| 10-19 | 49 | 0.0694 | 0.4118 | +0.3424 |
| 20-49 | 30 | 0.0355 | 0.3099 | +0.2744 |
| 50+ | 2 | 0.0306 | 0.3033 | +0.2727 |

### Every metric by fragment count

| Metric | 2-4 | 5-9 | 10-19 | 20-49 | 50+ |
| --- | --- | --- | --- | --- | --- |
| Exact Match | 1.0000 | 0.2857 | 0.0000 | 0.0000 | 0.0000 |
| Sequence Similarity | 1.0000 | 0.9082 | 0.4481 | 0.1805 | 0.1106 |
| Edit Similarity | 1.0000 | 0.8212 | 0.5263 | 0.3640 | 0.2730 |
| Adjacent Pair Accuracy | 1.0000 | 0.5857 | 0.4118 | 0.3099 | 0.3033 |
| Longest Correct Run | 1.0000 | 0.5212 | 0.2556 | 0.1308 | 0.0694 |
| Kendall Tau | 1.0000 | 0.7838 | 0.3813 | 0.1901 | 0.0335 |

### N-terminal start

| Measurement | Value |
| --- | --- |
| P(correct N-terminal start) | 0.8600 |
| ...on shuffled orderings | 0.0700 |
| Exact Match \| correct start | 0.1047 (9/86) |
| Exact Match \| wrong start | 0.0000 (0/14) |

Exact reconstruction is effectively conditional on getting the first fragment right — an ordering that starts wrong has already displaced every fragment after it.

### Breakpoints

| Measurement | Value |
| --- | --- |
| Samples | 100 |
| Mean breakpoints per protein | 10.70 |
| Median | 9.00 |
| Min / Max | 0 / 36 |
| Mean breakpoints per join | 0.5672 |
| Proteins assembled with 0 breakpoints | 9 |

### Error taxonomy

Failures are classified from the stored metric values (which were computed with the correct fragment-string semantics), checked in a fixed order so each sample lands in exactly one class. The cut points are disclosed in the table note and affect labelling only — no headline number depends on them.

| Failure mode | Samples | Share |
| --- | --- | --- |
| Exact reconstruction | 9 | 9.0% |
| Wrong start (structured, misanchored) | 14 | 14.0% |
| Partial assembly (correct start) | 77 | 77.0% |

## G. Cost

| Measurement | Value |
| --- | --- |
| LLM model | gpt-5-mini |
| Total LLM calls | 400 |
| Total LLM tokens | 595999 |
| LLM calls per sample | 4.00 |
| LLM tokens per sample | 5960.0 |
| Lever-choice failures | 0 |
| Wall clock per sample (total) | 43.8 s |
| Wall clock per sample (agentic arm) | 37.8 s |
| Wall clock per sample (control arm) | 5.8 s |
| LLM-Guided / Random Search time ratio | 6.49 |
| Completed samples | 100/100 |
| True order recovered | 100/100 |

The agentic arm costs **6.5x** the control arm's wall clock and **4.0 LLM calls per sample**, against which section D's paired tests are the return.

## Provenance

- Source: `samples.jsonl` in this folder — per-sample stored data only.
- No GPU, no model loading and no network access were used to build this report.
- Regenerate with `python -m evaluation.rebuild --run 140726_144552_agentic`.
- `report.md` (the run's original report) is left untouched; this file is additive.
