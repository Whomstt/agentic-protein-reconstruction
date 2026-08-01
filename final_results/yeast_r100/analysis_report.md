# Agentic Evaluation (Yeast, esm_small, r100) — Statistical Report

_Generated 2026-08-01 18:02 by `python -m evaluation.rebuild`. Every number is computed from `samples.jsonl` (n=100); nothing is transcribed by hand. Bootstraps use a fixed seed, so a rebuild reproduces this file exactly._

| Setting | Value |
| --- | --- |
| Organism | Yeast |
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
| Exact Match | 0.0000 [0.0000, 0.0370] | 0.1600 [0.1010, 0.2442] | 0.2600 [0.1840, 0.3537] | 0.2600 [0.1840, 0.3537] | 0.3500 [0.2636, 0.4475] |
| Sequence Similarity | 0.2128 [0.1645, 0.2710] | 0.5095 [0.4448, 0.5750] | 0.5485 [0.4758, 0.6203] | 0.5474 [0.4771, 0.6181] | 0.6474 [0.5743, 0.7166] |
| Edit Similarity | n/a | 0.5817 [0.5364, 0.6325] | 0.6333 [0.5820, 0.6861] | 0.6411 [0.5895, 0.6947] | 0.7509 [0.7022, 0.7968] |
| Adjacent Pair Accuracy | 0.0502 [0.0382, 0.0655] | 0.7034 [0.6673, 0.7406] | 0.7876 [0.7564, 0.8188] | 0.8118 [0.7832, 0.8384] | 0.8546 [0.8265, 0.8798] |
| Longest Correct Run | 0.0937 [0.0786, 0.1165] | 0.4479 [0.3965, 0.5104] | 0.5128 [0.4525, 0.5793] | 0.5297 [0.4696, 0.5943] | 0.6226 [0.5610, 0.6849] |
| Kendall Tau | -0.0308 [-0.0866, 0.0145] | 0.5005 [0.4343, 0.5689] | 0.5984 [0.5302, 0.6640] | 0.6241 [0.5599, 0.6863] | 0.7328 [0.6739, 0.7866] |

## B. Method Ladder

| Metric | Random Order | Fixed Settings | Random Search (no LLM) | LLM-Guided | Best Candidate (ceiling) |
| --- | --- | --- | --- | --- | --- |
| Exact Match | 0.0000 | 0.1600 (+0.1600) | 0.2600 (+0.1000) | 0.2600 (+0.0000) | 0.3500 (+0.0900) |
| Sequence Similarity | 0.2128 | 0.5095 (+0.2967) | 0.5485 (+0.0390) | 0.5474 (-0.0011) | 0.6474 (+0.1000) |
| Edit Similarity | n/a | 0.5817 (n/a) | 0.6333 (+0.0517) | 0.6411 (+0.0078) | 0.7509 (+0.1098) |
| Adjacent Pair Accuracy | 0.0502 | 0.7034 (+0.6532) | 0.7876 (+0.0842) | 0.8118 (+0.0241) | 0.8546 (+0.0428) |
| Longest Correct Run | 0.0937 | 0.4479 (+0.3542) | 0.5128 (+0.0649) | 0.5297 (+0.0169) | 0.6226 (+0.0929) |
| Kendall Tau | -0.0308 | 0.5005 (+0.5313) | 0.5984 (+0.0979) | 0.6241 (+0.0257) | 0.7328 (+0.1087) |

### Reading the Sequence Similarity floor

**A random shuffle of the fragments already scores 0.2128 on Sequence Similarity.** Every candidate ordering is a permutation of the *same* fragment multiset, so the string composition is identical across all arms and only the order varies. `difflib.SequenceMatcher` credits matching blocks wherever they occur, so a large fraction of that ratio is bought by composition alone and is available to a method that has learned nothing.

For contrast, on the same shuffled orderings the ordering-sensitive metrics sit at their true floor: Exact Match 0.0000, Adjacent Pair Accuracy 0.0502, Kendall Tau -0.0308. **Read Sequence Similarity only as a delta against the shuffled floor, never as an absolute.** Adjacent Pair Accuracy is the primary ordering metric.

## C. Replica Scaling

This run sits at **100 digestion replicas**. Replica count is what determines how many adjacencies the overlap graph can confirm outright, which is near-ground-truth structural information the search gets for free.

| Quantity | Value |
| --- | --- |
| Digestion replicas | 100 |
| Mean confirmed adjacencies per protein | 30.75 |
| Mean true joins covered by the overlap graph | 0.9610 |
| Mean junctions pruned by trypsin filter (%) | 5.02 |

_The scaling curve across replica counts is in the cross-run report_ (`cross_run_report.md`), _which needs more than one run to draw._

## D. Isolating the LLM's Contribution

The LLM-Guided and Random Search arms run on the **same proteins** with the same iteration budget, the same tool pipeline and the same best-validity selection; only the source of the five lever values differs (LLM vs. a non-LLM policy). They are therefore **paired**, and the comparison uses paired tests rather than asking whether the two arms' confidence intervals overlap — an overlap test on paired data is both wrong and badly underpowered.

Exact Match uses an **exact McNemar test** on the discordant pairs; the four continuous metrics use a **Wilcoxon signed-rank test**. Because five metrics are tested on one hypothesis, p-values are corrected with **Holm** across the family.

### LLM-Guided − Random Search

| Metric | Mean Δ | 95% CI | Test | Discordant / non-zero pairs | p | p (Holm) | Significant |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Exact Match | +0.0000 | [+0.0000, +0.0000] | McNemar (exact) | 0 (A only 0, B only 0) | 1.000 | 1.000 | no |
| Sequence Similarity | -0.0011 | [-0.0288, +0.0196] | Wilcoxon | 43 non-zero (26+ / 17-) | 0.398 | 0.797 | no |
| Edit Similarity | +0.0078 | [-0.0158, +0.0312] | Wilcoxon | 56 non-zero (36+ / 20-) | 0.259 | 0.792 | no |
| Adjacent Pair Accuracy | +0.0241 | [+0.0111, +0.0394] | Wilcoxon | 48 non-zero (33+ / 15-) | 0.001 | 0.009 | yes |
| Longest Correct Run | +0.0169 | [+0.0022, +0.0369] | Wilcoxon | 34 non-zero (22+ / 12-) | 0.061 | 0.306 | no |
| Kendall Tau | +0.0257 | [-0.0040, +0.0646] | Wilcoxon | 57 non-zero (33+ / 24-) | 0.198 | 0.792 | no |

Significant after Holm correction: **Adjacent Pair Accuracy**.

### LLM-Guided − Fixed Settings

| Metric | Mean Δ | 95% CI | Test | Discordant / non-zero pairs | p | p (Holm) | Significant |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Exact Match | +0.1000 | [+0.0400, +0.1600] | McNemar (exact) | 10 (A only 10, B only 0) | 0.002 | 0.005 | yes |
| Sequence Similarity | +0.0379 | [+0.0061, +0.0691] | Wilcoxon | 61 non-zero (39+ / 22-) | 0.016 | 0.016 | yes |
| Edit Similarity | +0.0595 | [+0.0262, +0.0970] | Wilcoxon | 70 non-zero (50+ / 20-) | 0.002 | 0.005 | yes |
| Adjacent Pair Accuracy | +0.1083 | [+0.0836, +0.1404] | Wilcoxon | 69 non-zero (63+ / 6-) | <0.001 | <0.001 | yes |
| Longest Correct Run | +0.0818 | [+0.0486, +0.1262] | Wilcoxon | 57 non-zero (42+ / 15-) | <0.001 | <0.001 | yes |
| Kendall Tau | +0.1236 | [+0.0715, +0.1883] | Wilcoxon | 70 non-zero (46+ / 24-) | <0.001 | 0.003 | yes |

Significant after Holm correction: **Exact Match, Sequence Similarity, Edit Similarity, Adjacent Pair Accuracy, Longest Correct Run, Kendall Tau**.

## E. Where the Bottleneck Is

### Junction scorer ranking (search-independent)

n/a - requires field: `junction_ranking` (per-sample top-1/top-3/MRR). The dense junction score matrix is not stored in samples.jsonl, and recomputing it needs pLM inference, which this offline rebuild does not do. Runs from the instrumentation change onward record it at zero extra model cost; for older runs, `python -m evaluation.junction_ranking` measures it separately.

### Selection signal trust

The run keeps whichever candidate scores best on the validity signal, so the signal's ability to rank candidates bounds what the search can deliver. **0.50 is a coin flip.**

| Measurement | Value |
| --- | --- |
| Samples with comparable candidate pairs | 81 |
| Comparable candidate pairs | 627 |
| Mean within-sample concordance | 0.6671 |
| Samples where concordance > 0.50 | 55 |
| Validity junction window | n/a |
| Validity confirmed penalty | n/a |

### Selection ceiling (Best Candidate)

| Metric | LLM-Guided | Best Candidate | Gap | Samples with a gap |
| --- | --- | --- | --- | --- |
| Exact Match | 0.2600 | 0.3500 | +0.0900 | 9/100 |
| Sequence Similarity | 0.5474 | 0.6474 | +0.1000 | 44/100 |
| Edit Similarity | 0.6411 | 0.7509 | +0.1098 | 49/100 |
| Adjacent Pair Accuracy | 0.8118 | 0.8546 | +0.0428 | 44/99 |
| Longest Correct Run | 0.5297 | 0.6226 | +0.0929 | 40/99 |
| Kendall Tau | 0.6241 | 0.7328 | +0.1087 | 51/99 |

On Adjacent Pair Accuracy the run leaves **0.0428** on the table in candidates it had already generated but did not select (44/99 samples). That is the size of the prize for a better selection signal alone.

### Trypsin filter recall

The filter pruned **5.02%** of candidate junctions on average. Whether any pruned junction was a *true* one is n/a - requires field: `trypsin_recall` (which junctions the trypsin filter pruned). Older runs stored only the pruned COUNT (`num_pruned`), which cannot tell us whether a pruned junction was a true one. Recorded from the instrumentation change onward.

## F. Difficulty Stratification and Error Modes

### Adjacent Pair Accuracy by fragment count

Difficulty scales with how many pieces the protein was cut into: the number of possible orderings grows factorially, while the pLM's evidence per junction does not improve. Lift over the Random Order floor is the honest read of whether the method is doing anything at each difficulty.

| Fragments | n | Random Order | LLM-Guided | Lift (paired) |
| --- | --- | --- | --- | --- |
| 2-4 | 4 | 0.1667 | 1.0000 | +0.8333 |
| 5-9 | 8 | 0.0387 | 0.9464 | +0.9077 |
| 10-19 | 22 | 0.0721 | 0.8753 | +0.8033 |
| 20-49 | 47 | 0.0406 | 0.7789 | +0.7383 |
| 50+ | 19 | 0.0279 | 0.7182 | +0.6902 |

### Every metric by fragment count

| Metric | 2-4 | 5-9 | 10-19 | 20-49 | 50+ |
| --- | --- | --- | --- | --- | --- |
| Exact Match | 1.0000 | 0.8750 | 0.5000 | 0.0851 | 0.0000 |
| Sequence Similarity | 1.0000 | 0.9448 | 0.7899 | 0.4596 | 0.2211 |
| Edit Similarity | 1.0000 | 0.9525 | 0.7771 | 0.5816 | 0.4243 |
| Adjacent Pair Accuracy | 1.0000 | 0.9464 | 0.8753 | 0.7789 | 0.7182 |
| Longest Correct Run | 1.0000 | 0.9531 | 0.7645 | 0.4370 | 0.1921 |
| Kendall Tau | 1.0000 | 0.9911 | 0.7754 | 0.5850 | 0.2946 |

### N-terminal start

| Measurement | Value |
| --- | --- |
| P(correct N-terminal start) | 0.9300 |
| ...on shuffled orderings | 0.0200 |
| Exact Match \| correct start | 0.2796 (26/93) |
| Exact Match \| wrong start | 0.0000 (0/7) |

Exact reconstruction is effectively conditional on getting the first fragment right — an ordering that starts wrong has already displaced every fragment after it.

### Breakpoints

| Measurement | Value |
| --- | --- |
| Samples | 99 |
| Mean breakpoints per protein | 7.44 |
| Median | 5.00 |
| Min / Max | 0 / 59 |
| Mean breakpoints per join | 0.1882 |
| Proteins assembled with 0 breakpoints | 26 |

### Error taxonomy

Failures are classified from the stored metric values (which were computed with the correct fragment-string semantics), checked in a fixed order so each sample lands in exactly one class. The cut points are disclosed in the table note and affect labelling only — no headline number depends on them.

| Failure mode | Samples | Share |
| --- | --- | --- |
| Exact reconstruction | 26 | 26.0% |
| Local transposition | 3 | 3.0% |
| Wrong start (structured, misanchored) | 7 | 7.0% |
| Partial assembly (correct start) | 63 | 63.0% |
| Unclassified (no ground-truth order) | 1 | 1.0% |

## G. Cost

| Measurement | Value |
| --- | --- |
| LLM model | gpt-5-mini |
| Total LLM calls | 400 |
| Total LLM tokens | 603129 |
| LLM calls per sample | 4.00 |
| LLM tokens per sample | 6031.3 |
| Lever-choice failures | 0 |
| Wall clock per sample (total) | 79.9 s |
| Wall clock per sample (agentic arm) | 60.4 s |
| Wall clock per sample (control arm) | 19.3 s |
| LLM-Guided / Random Search time ratio | 3.12 |
| Completed samples | 100/100 |
| True order recovered | 99/100 |

The agentic arm costs **3.1x** the control arm's wall clock and **4.0 LLM calls per sample**, against which section D's paired tests are the return.

## Provenance

- Source: `samples.jsonl` in this folder — per-sample stored data only.
- No GPU, no model loading and no network access were used to build this report.
- Regenerate with `python -m evaluation.rebuild --run 140726_171539_agentic`.
- `report.md` (the run's original report) is left untouched; this file is additive.
