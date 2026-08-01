# Agentic Evaluation (Yeast, esm_small, r20) — Statistical Report

_Generated 2026-08-01 18:02 by `python -m evaluation.rebuild`. Every number is computed from `samples.jsonl` (n=100); nothing is transcribed by hand. Bootstraps use a fixed seed, so a rebuild reproduces this file exactly._

| Setting | Value |
| --- | --- |
| Organism | Yeast |
| Digestion replicas | 20 |
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
| Exact Match | 0.0100 [0.0018, 0.0545] | 0.1000 [0.0552, 0.1744] | 0.1500 [0.0931, 0.2328] | 0.1600 [0.1010, 0.2442] | 0.2300 [0.1584, 0.3215] |
| Sequence Similarity | 0.2178 [0.1683, 0.2764] | 0.3682 [0.3051, 0.4380] | 0.4005 [0.3363, 0.4742] | 0.4105 [0.3431, 0.4844] | 0.4671 [0.3986, 0.5421] |
| Edit Similarity | n/a | 0.4627 [0.4201, 0.5146] | 0.5061 [0.4592, 0.5622] | 0.5183 [0.4697, 0.5755] | 0.5996 [0.5502, 0.6556] |
| Adjacent Pair Accuracy | 0.0640 [0.0510, 0.0819] | 0.4992 [0.4621, 0.5434] | 0.6161 [0.5770, 0.6576] | 0.6417 [0.6043, 0.6811] | 0.6923 [0.6532, 0.7324] |
| Longest Correct Run | 0.1076 [0.0898, 0.1410] | 0.3003 [0.2533, 0.3599] | 0.3613 [0.3091, 0.4260] | 0.3772 [0.3229, 0.4407] | 0.4425 [0.3814, 0.5116] |
| Kendall Tau | 0.0444 [0.0114, 0.0801] | 0.3240 [0.2647, 0.3897] | 0.4285 [0.3667, 0.4994] | 0.4500 [0.3857, 0.5203] | 0.5468 [0.4857, 0.6103] |

## B. Method Ladder

| Metric | Random Order | Fixed Settings | Random Search (no LLM) | LLM-Guided | Best Candidate (ceiling) |
| --- | --- | --- | --- | --- | --- |
| Exact Match | 0.0100 | 0.1000 (+0.0900) | 0.1500 (+0.0500) | 0.1600 (+0.0100) | 0.2300 (+0.0700) |
| Sequence Similarity | 0.2178 | 0.3682 (+0.1504) | 0.4005 (+0.0322) | 0.4105 (+0.0101) | 0.4671 (+0.0566) |
| Edit Similarity | n/a | 0.4627 (n/a) | 0.5061 (+0.0433) | 0.5183 (+0.0123) | 0.5996 (+0.0813) |
| Adjacent Pair Accuracy | 0.0640 | 0.4992 (+0.4352) | 0.6161 (+0.1169) | 0.6417 (+0.0256) | 0.6923 (+0.0507) |
| Longest Correct Run | 0.1076 | 0.3003 (+0.1927) | 0.3613 (+0.0610) | 0.3772 (+0.0159) | 0.4425 (+0.0652) |
| Kendall Tau | 0.0444 | 0.3240 (+0.2796) | 0.4285 (+0.1045) | 0.4500 (+0.0215) | 0.5468 (+0.0968) |

### Reading the Sequence Similarity floor

**A random shuffle of the fragments already scores 0.2178 on Sequence Similarity.** Every candidate ordering is a permutation of the *same* fragment multiset, so the string composition is identical across all arms and only the order varies. `difflib.SequenceMatcher` credits matching blocks wherever they occur, so a large fraction of that ratio is bought by composition alone and is available to a method that has learned nothing.

For contrast, on the same shuffled orderings the ordering-sensitive metrics sit at their true floor: Exact Match 0.0100, Adjacent Pair Accuracy 0.0640, Kendall Tau 0.0444. **Read Sequence Similarity only as a delta against the shuffled floor, never as an absolute.** Adjacent Pair Accuracy is the primary ordering metric.

## C. Replica Scaling

This run sits at **20 digestion replicas**. Replica count is what determines how many adjacencies the overlap graph can confirm outright, which is near-ground-truth structural information the search gets for free.

| Quantity | Value |
| --- | --- |
| Digestion replicas | 20 |
| Mean confirmed adjacencies per protein | 23.24 |
| Mean true joins covered by the overlap graph | 0.7304 |
| Mean junctions pruned by trypsin filter (%) | 5.08 |

_The scaling curve across replica counts is in the cross-run report_ (`cross_run_report.md`), _which needs more than one run to draw._

## D. Isolating the LLM's Contribution

The LLM-Guided and Random Search arms run on the **same proteins** with the same iteration budget, the same tool pipeline and the same best-validity selection; only the source of the five lever values differs (LLM vs. a non-LLM policy). They are therefore **paired**, and the comparison uses paired tests rather than asking whether the two arms' confidence intervals overlap — an overlap test on paired data is both wrong and badly underpowered.

Exact Match uses an **exact McNemar test** on the discordant pairs; the four continuous metrics use a **Wilcoxon signed-rank test**. Because five metrics are tested on one hypothesis, p-values are corrected with **Holm** across the family.

### LLM-Guided − Random Search

| Metric | Mean Δ | 95% CI | Test | Discordant / non-zero pairs | p | p (Holm) | Significant |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Exact Match | +0.0100 | [+0.0000, +0.0300] | McNemar (exact) | 1 (A only 1, B only 0) | 1.000 | 1.000 | no |
| Sequence Similarity | +0.0101 | [-0.0075, +0.0313] | Wilcoxon | 42 non-zero (23+ / 19-) | 0.303 | 1.000 | no |
| Edit Similarity | +0.0123 | [-0.0067, +0.0360] | Wilcoxon | 66 non-zero (34+ / 32-) | 0.523 | 1.000 | no |
| Adjacent Pair Accuracy | +0.0256 | [+0.0098, +0.0445] | Wilcoxon | 58 non-zero (39+ / 19-) | 0.009 | 0.053 | no |
| Longest Correct Run | +0.0159 | [+0.0005, +0.0396] | Wilcoxon | 31 non-zero (19+ / 12-) | 0.167 | 0.835 | no |
| Kendall Tau | +0.0215 | [-0.0073, +0.0547] | Wilcoxon | 67 non-zero (37+ / 30-) | 0.261 | 1.000 | no |

**No metric survives Holm correction.** The observed differences are consistent with what re-running the same non-LLM policy on these n=100 proteins could produce by chance; this run does not demonstrate a reasoning advantage on this comparison.

### LLM-Guided − Fixed Settings

| Metric | Mean Δ | 95% CI | Test | Discordant / non-zero pairs | p | p (Holm) | Significant |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Exact Match | +0.0600 | [+0.0100, +0.1100] | McNemar (exact) | 8 (A only 7, B only 1) | 0.070 | 0.070 | no |
| Sequence Similarity | +0.0423 | [+0.0164, +0.0729] | Wilcoxon | 72 non-zero (46+ / 26-) | 0.003 | 0.007 | yes |
| Edit Similarity | +0.0556 | [+0.0263, +0.0908] | Wilcoxon | 83 non-zero (58+ / 25-) | 0.001 | 0.003 | yes |
| Adjacent Pair Accuracy | +0.1425 | [+0.1146, +0.1718] | Wilcoxon | 79 non-zero (75+ / 4-) | <0.001 | <0.001 | yes |
| Longest Correct Run | +0.0769 | [+0.0523, +0.1091] | Wilcoxon | 68 non-zero (59+ / 9-) | <0.001 | <0.001 | yes |
| Kendall Tau | +0.1260 | [+0.0788, +0.1818] | Wilcoxon | 82 non-zero (56+ / 26-) | <0.001 | <0.001 | yes |

Significant after Holm correction: **Sequence Similarity, Edit Similarity, Adjacent Pair Accuracy, Longest Correct Run, Kendall Tau**.

## E. Where the Bottleneck Is

### Junction scorer ranking (search-independent)

n/a - requires field: `junction_ranking` (per-sample top-1/top-3/MRR). The dense junction score matrix is not stored in samples.jsonl, and recomputing it needs pLM inference, which this offline rebuild does not do. Runs from the instrumentation change onward record it at zero extra model cost; for older runs, `python -m evaluation.junction_ranking` measures it separately.

### Selection signal trust

The run keeps whichever candidate scores best on the validity signal, so the signal's ability to rank candidates bounds what the search can deliver. **0.50 is a coin flip.**

| Measurement | Value |
| --- | --- |
| Samples with comparable candidate pairs | 91 |
| Comparable candidate pairs | 716 |
| Mean within-sample concordance | 0.7511 |
| Samples where concordance > 0.50 | 77 |
| Validity junction window | n/a |
| Validity confirmed penalty | n/a |

### Selection ceiling (Best Candidate)

| Metric | LLM-Guided | Best Candidate | Gap | Samples with a gap |
| --- | --- | --- | --- | --- |
| Exact Match | 0.1600 | 0.2300 | +0.0700 | 7/100 |
| Sequence Similarity | 0.4105 | 0.4671 | +0.0566 | 54/100 |
| Edit Similarity | 0.5183 | 0.5996 | +0.0813 | 60/100 |
| Adjacent Pair Accuracy | 0.6417 | 0.6923 | +0.0507 | 43/100 |
| Longest Correct Run | 0.3772 | 0.4425 | +0.0652 | 34/100 |
| Kendall Tau | 0.4500 | 0.5468 | +0.0968 | 59/100 |

On Adjacent Pair Accuracy the run leaves **0.0507** on the table in candidates it had already generated but did not select (43/100 samples). That is the size of the prize for a better selection signal alone.

### Trypsin filter recall

The filter pruned **5.08%** of candidate junctions on average. Whether any pruned junction was a *true* one is n/a - requires field: `trypsin_recall` (which junctions the trypsin filter pruned). Older runs stored only the pruned COUNT (`num_pruned`), which cannot tell us whether a pruned junction was a true one. Recorded from the instrumentation change onward.

## F. Difficulty Stratification and Error Modes

### Adjacent Pair Accuracy by fragment count

Difficulty scales with how many pieces the protein was cut into: the number of possible orderings grows factorially, while the pLM's evidence per junction does not improve. Lift over the Random Order floor is the honest read of whether the method is doing anything at each difficulty.

| Fragments | n | Random Order | LLM-Guided | Lift (paired) |
| --- | --- | --- | --- | --- |
| 2-4 | 1 | 0.0000 | 1.0000 | +1.0000 |
| 5-9 | 17 | 0.1004 | 0.8609 | +0.7606 |
| 10-19 | 19 | 0.1222 | 0.6965 | +0.5743 |
| 20-49 | 43 | 0.0462 | 0.5929 | +0.5467 |
| 50+ | 19 | 0.0200 | 0.5159 | +0.4959 |

### Every metric by fragment count

| Metric | 2-4 | 5-9 | 10-19 | 20-49 | 50+ |
| --- | --- | --- | --- | --- | --- |
| Exact Match | 1.0000 | 0.7059 | 0.1053 | 0.0000 | 0.0000 |
| Sequence Similarity | 1.0000 | 0.9351 | 0.6381 | 0.2269 | 0.0672 |
| Edit Similarity | 1.0000 | 0.8767 | 0.6254 | 0.4130 | 0.2782 |
| Adjacent Pair Accuracy | 1.0000 | 0.8609 | 0.6965 | 0.5929 | 0.5159 |
| Longest Correct Run | 1.0000 | 0.8242 | 0.4731 | 0.2429 | 0.1200 |
| Kendall Tau | 1.0000 | 0.9000 | 0.6038 | 0.3126 | 0.1990 |

### N-terminal start

| Measurement | Value |
| --- | --- |
| P(correct N-terminal start) | 0.9100 |
| ...on shuffled orderings | 0.0800 |
| Exact Match \| correct start | 0.1758 (16/91) |
| Exact Match \| wrong start | 0.0000 (0/9) |

Exact reconstruction is effectively conditional on getting the first fragment right — an ordering that starts wrong has already displaced every fragment after it.

### Breakpoints

| Measurement | Value |
| --- | --- |
| Samples | 99 |
| Mean breakpoints per protein | 13.94 |
| Median | 10.00 |
| Min / Max | 0 / 117 |
| Mean breakpoints per join | 0.3518 |
| Proteins assembled with 0 breakpoints | 15 |

### Error taxonomy

Failures are classified from the stored metric values (which were computed with the correct fragment-string semantics), checked in a fixed order so each sample lands in exactly one class. The cut points are disclosed in the table note and affect labelling only — no headline number depends on them.

| Failure mode | Samples | Share |
| --- | --- | --- |
| Exact reconstruction | 16 | 16.0% |
| Local transposition | 1 | 1.0% |
| Wrong start (structured, misanchored) | 9 | 9.0% |
| Partial assembly (correct start) | 74 | 74.0% |

## G. Cost

| Measurement | Value |
| --- | --- |
| LLM model | gpt-5-mini |
| Total LLM calls | 400 |
| Total LLM tokens | 596101 |
| LLM calls per sample | 4.00 |
| LLM tokens per sample | 5961.0 |
| Lever-choice failures | 0 |
| Wall clock per sample (total) | 82.7 s |
| Wall clock per sample (agentic arm) | 63.1 s |
| Wall clock per sample (control arm) | 19.5 s |
| LLM-Guided / Random Search time ratio | 3.24 |
| Completed samples | 100/100 |
| True order recovered | 100/100 |

The agentic arm costs **3.2x** the control arm's wall clock and **4.0 LLM calls per sample**, against which section D's paired tests are the return.

## Provenance

- Source: `samples.jsonl` in this folder — per-sample stored data only.
- No GPU, no model loading and no network access were used to build this report.
- Regenerate with `python -m evaluation.rebuild --run 140726_194805_agentic`.
- `report.md` (the run's original report) is left untouched; this file is additive.
