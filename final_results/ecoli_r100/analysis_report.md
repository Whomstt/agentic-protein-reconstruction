# Agentic Evaluation (E. coli, esm_small, r100) — Statistical Report

_Generated 2026-08-01 18:01 by `python -m evaluation.rebuild`. Every number is computed from `samples.jsonl` (n=100); nothing is transcribed by hand. Bootstraps use a fixed seed, so a rebuild reproduces this file exactly._

| Setting | Value |
| --- | --- |
| Organism | E. coli |
| Digestion replicas | 100 |
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
| Exact Match | 0.0100 [0.0018, 0.0545] | 0.3200 [0.2367, 0.4166] | 0.4000 [0.3094, 0.4980] | 0.4600 [0.3656, 0.5574] | 0.6400 [0.5424, 0.7273] |
| Sequence Similarity | 0.2888 [0.2349, 0.3462] | 0.6577 [0.5869, 0.7271] | 0.7084 [0.6381, 0.7709] | 0.7651 [0.6999, 0.8220] | 0.8516 [0.7981, 0.8950] |
| Edit Similarity | n/a | 0.7251 [0.6710, 0.7786] | 0.7507 [0.6962, 0.8021] | 0.7954 [0.7445, 0.8425] | 0.8861 [0.8449, 0.9211] |
| Adjacent Pair Accuracy | 0.0676 [0.0506, 0.0916] | 0.7841 [0.7439, 0.8194] | 0.8627 [0.8287, 0.8879] | 0.8791 [0.8451, 0.9037] | 0.9256 [0.8906, 0.9458] |
| Longest Correct Run | 0.1362 [0.1146, 0.1698] | 0.6283 [0.5707, 0.6898] | 0.6819 [0.6246, 0.7387] | 0.7169 [0.6595, 0.7734] | 0.8195 [0.7685, 0.8672] |
| Kendall Tau | -0.0064 [-0.0580, 0.0382] | 0.6315 [0.5512, 0.7050] | 0.7050 [0.6325, 0.7681] | 0.7532 [0.6806, 0.8114] | 0.8788 [0.8279, 0.9159] |

## B. Method Ladder

| Metric | Random Order | Fixed Settings | Random Search (no LLM) | LLM-Guided | Best Candidate (ceiling) |
| --- | --- | --- | --- | --- | --- |
| Exact Match | 0.0100 | 0.3200 (+0.3100) | 0.4000 (+0.0800) | 0.4600 (+0.0600) | 0.6400 (+0.1800) |
| Sequence Similarity | 0.2888 | 0.6577 (+0.3689) | 0.7084 (+0.0507) | 0.7651 (+0.0567) | 0.8516 (+0.0865) |
| Edit Similarity | n/a | 0.7251 (n/a) | 0.7507 (+0.0256) | 0.7954 (+0.0447) | 0.8861 (+0.0907) |
| Adjacent Pair Accuracy | 0.0676 | 0.7841 (+0.7166) | 0.8627 (+0.0786) | 0.8791 (+0.0164) | 0.9256 (+0.0465) |
| Longest Correct Run | 0.1362 | 0.6283 (+0.4921) | 0.6819 (+0.0536) | 0.7169 (+0.0350) | 0.8195 (+0.1026) |
| Kendall Tau | -0.0064 | 0.6315 (+0.6379) | 0.7050 (+0.0735) | 0.7532 (+0.0482) | 0.8788 (+0.1256) |

### Reading the Sequence Similarity floor

**A random shuffle of the fragments already scores 0.2888 on Sequence Similarity.** Every candidate ordering is a permutation of the *same* fragment multiset, so the string composition is identical across all arms and only the order varies. `difflib.SequenceMatcher` credits matching blocks wherever they occur, so a large fraction of that ratio is bought by composition alone and is available to a method that has learned nothing.

For contrast, on the same shuffled orderings the ordering-sensitive metrics sit at their true floor: Exact Match 0.0100, Adjacent Pair Accuracy 0.0676, Kendall Tau -0.0064. **Read Sequence Similarity only as a delta against the shuffled floor, never as an absolute.** Adjacent Pair Accuracy is the primary ordering metric.

## C. Replica Scaling

This run sits at **100 digestion replicas**. Replica count is what determines how many adjacencies the overlap graph can confirm outright, which is near-ground-truth structural information the search gets for free.

| Quantity | Value |
| --- | --- |
| Digestion replicas | 100 |
| Mean confirmed adjacencies per protein | 18.20 |
| Mean true joins covered by the overlap graph | 0.9346 |
| Mean junctions pruned by trypsin filter (%) | 6.58 |

_The scaling curve across replica counts is in the cross-run report_ (`cross_run_report.md`), _which needs more than one run to draw._

## D. Isolating the LLM's Contribution

The LLM-Guided and Random Search arms run on the **same proteins** with the same iteration budget, the same tool pipeline and the same best-validity selection; only the source of the five lever values differs (LLM vs. a non-LLM policy). They are therefore **paired**, and the comparison uses paired tests rather than asking whether the two arms' confidence intervals overlap — an overlap test on paired data is both wrong and badly underpowered.

Exact Match uses an **exact McNemar test** on the discordant pairs; the four continuous metrics use a **Wilcoxon signed-rank test**. Because five metrics are tested on one hypothesis, p-values are corrected with **Holm** across the family.

### LLM-Guided − Random Search

| Metric | Mean Δ | 95% CI | Test | Discordant / non-zero pairs | p | p (Holm) | Significant |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Exact Match | +0.0600 | [+0.0000, +0.1100] | McNemar (exact) | 8 (A only 7, B only 1) | 0.070 | 0.078 | no |
| Sequence Similarity | +0.0567 | [+0.0195, +0.1096] | Wilcoxon | 25 non-zero (17+ / 8-) | 0.026 | 0.078 | no |
| Edit Similarity | +0.0447 | [+0.0181, +0.0838] | Wilcoxon | 31 non-zero (23+ / 8-) | 0.004 | 0.022 | yes |
| Adjacent Pair Accuracy | +0.0164 | [+0.0044, +0.0309] | Wilcoxon | 28 non-zero (21+ / 7-) | 0.014 | 0.070 | no |
| Longest Correct Run | +0.0350 | [+0.0118, +0.0687] | Wilcoxon | 26 non-zero (18+ / 8-) | 0.030 | 0.078 | no |
| Kendall Tau | +0.0482 | [+0.0182, +0.0960] | Wilcoxon | 34 non-zero (24+ / 10-) | 0.020 | 0.078 | no |

Significant after Holm correction: **Edit Similarity**.

### LLM-Guided − Fixed Settings

| Metric | Mean Δ | 95% CI | Test | Discordant / non-zero pairs | p | p (Holm) | Significant |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Exact Match | +0.1400 | [+0.0600, +0.2100] | McNemar (exact) | 16 (A only 15, B only 1) | <0.001 | 0.001 | yes |
| Sequence Similarity | +0.1074 | [+0.0584, +0.1641] | Wilcoxon | 41 non-zero (34+ / 7-) | <0.001 | <0.001 | yes |
| Edit Similarity | +0.0703 | [+0.0298, +0.1160] | Wilcoxon | 54 non-zero (37+ / 17-) | <0.001 | 0.001 | yes |
| Adjacent Pair Accuracy | +0.0949 | [+0.0682, +0.1289] | Wilcoxon | 51 non-zero (46+ / 5-) | <0.001 | <0.001 | yes |
| Longest Correct Run | +0.0886 | [+0.0459, +0.1370] | Wilcoxon | 46 non-zero (34+ / 12-) | <0.001 | <0.001 | yes |
| Kendall Tau | +0.1218 | [+0.0541, +0.1927] | Wilcoxon | 56 non-zero (42+ / 14-) | <0.001 | <0.001 | yes |

Significant after Holm correction: **Exact Match, Sequence Similarity, Edit Similarity, Adjacent Pair Accuracy, Longest Correct Run, Kendall Tau**.

## E. Where the Bottleneck Is

### Junction scorer ranking (search-independent)

n/a - requires field: `junction_ranking` (per-sample top-1/top-3/MRR). The dense junction score matrix is not stored in samples.jsonl, and recomputing it needs pLM inference, which this offline rebuild does not do. Runs from the instrumentation change onward record it at zero extra model cost; for older runs, `python -m evaluation.junction_ranking` measures it separately.

### Selection signal trust

The run keeps whichever candidate scores best on the validity signal, so the signal's ability to rank candidates bounds what the search can deliver. **0.50 is a coin flip.**

| Measurement | Value |
| --- | --- |
| Samples with comparable candidate pairs | 68 |
| Comparable candidate pairs | 476 |
| Mean within-sample concordance | 0.6371 |
| Samples where concordance > 0.50 | 46 |
| Validity junction window | n/a |
| Validity confirmed penalty | n/a |

### Selection ceiling (Best Candidate)

| Metric | LLM-Guided | Best Candidate | Gap | Samples with a gap |
| --- | --- | --- | --- | --- |
| Exact Match | 0.4600 | 0.6400 | +0.1800 | 18/100 |
| Sequence Similarity | 0.7651 | 0.8516 | +0.0865 | 37/100 |
| Edit Similarity | 0.7954 | 0.8861 | +0.0907 | 41/100 |
| Adjacent Pair Accuracy | 0.8791 | 0.9256 | +0.0465 | 35/100 |
| Longest Correct Run | 0.7169 | 0.8195 | +0.1026 | 37/100 |
| Kendall Tau | 0.7532 | 0.8788 | +0.1256 | 40/100 |

On Adjacent Pair Accuracy the run leaves **0.0465** on the table in candidates it had already generated but did not select (35/100 samples). That is the size of the prize for a better selection signal alone.

### Trypsin filter recall

The filter pruned **6.58%** of candidate junctions on average. Whether any pruned junction was a *true* one is n/a - requires field: `trypsin_recall` (which junctions the trypsin filter pruned). Older runs stored only the pruned COUNT (`num_pruned`), which cannot tell us whether a pruned junction was a true one. Recorded from the instrumentation change onward.

## F. Difficulty Stratification and Error Modes

### Adjacent Pair Accuracy by fragment count

Difficulty scales with how many pieces the protein was cut into: the number of possible orderings grows factorially, while the pLM's evidence per junction does not improve. Lift over the Random Order floor is the honest read of whether the method is doing anything at each difficulty.

| Fragments | n | Random Order | LLM-Guided | Lift (paired) |
| --- | --- | --- | --- | --- |
| 2-4 | 3 | 0.0000 | 1.0000 | +1.0000 |
| 5-9 | 20 | 0.1309 | 0.9170 | +0.7861 |
| 10-19 | 39 | 0.0707 | 0.8967 | +0.8260 |
| 20-49 | 28 | 0.0415 | 0.8722 | +0.8307 |
| 50+ | 9 | 0.0248 | 0.7975 | +0.7727 |

### Every metric by fragment count

| Metric | 2-4 | 5-9 | 10-19 | 20-49 | 50+ |
| --- | --- | --- | --- | --- | --- |
| Exact Match | 1.0000 | 0.7500 | 0.4359 | 0.3571 | 0.0000 |
| Sequence Similarity | 1.0000 | 0.9652 | 0.8365 | 0.6040 | 0.4076 |
| Edit Similarity | 1.0000 | 0.9362 | 0.8456 | 0.7003 | 0.4699 |
| Adjacent Pair Accuracy | 1.0000 | 0.9170 | 0.8967 | 0.8722 | 0.7975 |
| Longest Correct Run | 1.0000 | 0.8931 | 0.7509 | 0.6216 | 0.3490 |
| Kendall Tau | 1.0000 | 0.9298 | 0.8471 | 0.5976 | 0.4398 |

### N-terminal start

| Measurement | Value |
| --- | --- |
| P(correct N-terminal start) | 0.9800 |
| ...on shuffled orderings | 0.1300 |
| Exact Match \| correct start | 0.4694 (46/98) |
| Exact Match \| wrong start | 0.0000 (0/2) |

Exact reconstruction is effectively conditional on getting the first fragment right — an ordering that starts wrong has already displaced every fragment after it.

### Breakpoints

| Measurement | Value |
| --- | --- |
| Samples | 99 |
| Mean breakpoints per protein | 2.71 |
| Median | 2.00 |
| Min / Max | 0 / 18 |
| Mean breakpoints per join | 0.1120 |
| Proteins assembled with 0 breakpoints | 46 |

### Error taxonomy

Failures are classified from the stored metric values (which were computed with the correct fragment-string semantics), checked in a fixed order so each sample lands in exactly one class. The cut points are disclosed in the table note and affect labelling only — no headline number depends on them.

| Failure mode | Samples | Share |
| --- | --- | --- |
| Exact reconstruction | 46 | 46.0% |
| Local transposition | 9 | 9.0% |
| Wrong start (structured, misanchored) | 2 | 2.0% |
| Partial assembly (correct start) | 43 | 43.0% |

## G. Cost

| Measurement | Value |
| --- | --- |
| LLM model | gpt-5-mini |
| Total LLM calls | 400 |
| Total LLM tokens | 613053 |
| LLM calls per sample | 4.00 |
| LLM tokens per sample | 6130.5 |
| Lever-choice failures | 0 |
| Wall clock per sample (total) | 43.0 s |
| Wall clock per sample (agentic arm) | 35.8 s |
| Wall clock per sample (control arm) | 7.0 s |
| LLM-Guided / Random Search time ratio | 5.09 |
| Completed samples | 100/100 |
| True order recovered | 100/100 |

The agentic arm costs **5.1x** the control arm's wall clock and **4.0 LLM calls per sample**, against which section D's paired tests are the return.

## Provenance

- Source: `samples.jsonl` in this folder — per-sample stored data only.
- No GPU, no model loading and no network access were used to build this report.
- Regenerate with `python -m evaluation.rebuild --run 130726_224804_agentic`.
- `report.md` (the run's original report) is left untouched; this file is additive.
