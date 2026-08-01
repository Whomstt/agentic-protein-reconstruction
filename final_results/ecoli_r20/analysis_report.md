# Agentic Evaluation (E. coli, esm_small, r20) — Statistical Report

_Generated 2026-08-01 18:01 by `python -m evaluation.rebuild`. Every number is computed from `samples.jsonl` (n=100); nothing is transcribed by hand. Bootstraps use a fixed seed, so a rebuild reproduces this file exactly._

| Setting | Value |
| --- | --- |
| Organism | E. coli |
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
| Exact Match | 0.0300 [0.0103, 0.0845] | 0.1700 [0.1089, 0.2555] | 0.2700 [0.1927, 0.3643] | 0.2900 [0.2101, 0.3854] | 0.3700 [0.2818, 0.4678] |
| Sequence Similarity | 0.3132 [0.2585, 0.3735] | 0.5051 [0.4369, 0.5713] | 0.5943 [0.5218, 0.6608] | 0.5768 [0.5021, 0.6505] | 0.6441 [0.5721, 0.7118] |
| Edit Similarity | n/a | 0.5693 [0.5189, 0.6223] | 0.6486 [0.5955, 0.7029] | 0.6622 [0.6087, 0.7152] | 0.7462 [0.6987, 0.7907] |
| Adjacent Pair Accuracy | 0.1199 [0.0894, 0.1679] | 0.5856 [0.5357, 0.6343] | 0.7100 [0.6672, 0.7505] | 0.7338 [0.6935, 0.7732] | 0.7826 [0.7439, 0.8197] |
| Longest Correct Run | 0.1701 [0.1399, 0.2155] | 0.4222 [0.3646, 0.4848] | 0.5021 [0.4364, 0.5695] | 0.5240 [0.4598, 0.5911] | 0.5821 [0.5146, 0.6505] |
| Kendall Tau | 0.0003 [-0.0547, 0.0604] | 0.4414 [0.3626, 0.5178] | 0.5890 [0.5212, 0.6541] | 0.6242 [0.5590, 0.6870] | 0.7240 [0.6674, 0.7761] |

## B. Method Ladder

| Metric | Random Order | Fixed Settings | Random Search (no LLM) | LLM-Guided | Best Candidate (ceiling) |
| --- | --- | --- | --- | --- | --- |
| Exact Match | 0.0300 | 0.1700 (+0.1400) | 0.2700 (+0.1000) | 0.2900 (+0.0200) | 0.3700 (+0.0800) |
| Sequence Similarity | 0.3132 | 0.5051 (+0.1919) | 0.5943 (+0.0892) | 0.5768 (-0.0175) | 0.6441 (+0.0673) |
| Edit Similarity | n/a | 0.5693 (n/a) | 0.6486 (+0.0793) | 0.6622 (+0.0135) | 0.7462 (+0.0840) |
| Adjacent Pair Accuracy | 0.1199 | 0.5856 (+0.4656) | 0.7100 (+0.1244) | 0.7338 (+0.0239) | 0.7826 (+0.0488) |
| Longest Correct Run | 0.1701 | 0.4222 (+0.2522) | 0.5021 (+0.0798) | 0.5240 (+0.0219) | 0.5821 (+0.0582) |
| Kendall Tau | 0.0003 | 0.4414 (+0.4412) | 0.5890 (+0.1476) | 0.6242 (+0.0352) | 0.7240 (+0.0997) |

### Reading the Sequence Similarity floor

**A random shuffle of the fragments already scores 0.3132 on Sequence Similarity.** Every candidate ordering is a permutation of the *same* fragment multiset, so the string composition is identical across all arms and only the order varies. `difflib.SequenceMatcher` credits matching blocks wherever they occur, so a large fraction of that ratio is bought by composition alone and is available to a method that has learned nothing.

For contrast, on the same shuffled orderings the ordering-sensitive metrics sit at their true floor: Exact Match 0.0300, Adjacent Pair Accuracy 0.1199, Kendall Tau 0.0003. **Read Sequence Similarity only as a delta against the shuffled floor, never as an absolute.** Adjacent Pair Accuracy is the primary ordering metric.

## C. Replica Scaling

This run sits at **20 digestion replicas**. Replica count is what determines how many adjacencies the overlap graph can confirm outright, which is near-ground-truth structural information the search gets for free.

| Quantity | Value |
| --- | --- |
| Digestion replicas | 20 |
| Mean confirmed adjacencies per protein | 12.95 |
| Mean true joins covered by the overlap graph | 0.6962 |
| Mean junctions pruned by trypsin filter (%) | 7.63 |

_The scaling curve across replica counts is in the cross-run report_ (`cross_run_report.md`), _which needs more than one run to draw._

## D. Isolating the LLM's Contribution

The LLM-Guided and Random Search arms run on the **same proteins** with the same iteration budget, the same tool pipeline and the same best-validity selection; only the source of the five lever values differs (LLM vs. a non-LLM policy). They are therefore **paired**, and the comparison uses paired tests rather than asking whether the two arms' confidence intervals overlap — an overlap test on paired data is both wrong and badly underpowered.

Exact Match uses an **exact McNemar test** on the discordant pairs; the four continuous metrics use a **Wilcoxon signed-rank test**. Because five metrics are tested on one hypothesis, p-values are corrected with **Holm** across the family.

### LLM-Guided − Random Search

| Metric | Mean Δ | 95% CI | Test | Discordant / non-zero pairs | p | p (Holm) | Significant |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Exact Match | +0.0200 | [-0.0200, +0.0500] | McNemar (exact) | 4 (A only 3, B only 1) | 0.625 | 0.625 | no |
| Sequence Similarity | -0.0175 | [-0.0439, +0.0116] | Wilcoxon | 30 non-zero (12+ / 18-) | 0.080 | 0.251 | no |
| Edit Similarity | +0.0135 | [-0.0059, +0.0341] | Wilcoxon | 47 non-zero (30+ / 17-) | 0.109 | 0.251 | no |
| Adjacent Pair Accuracy | +0.0239 | [+0.0103, +0.0409] | Wilcoxon | 39 non-zero (28+ / 11-) | 0.001 | 0.009 | yes |
| Longest Correct Run | +0.0219 | [+0.0056, +0.0500] | Wilcoxon | 24 non-zero (17+ / 7-) | 0.061 | 0.251 | no |
| Kendall Tau | +0.0352 | [+0.0067, +0.0705] | Wilcoxon | 48 non-zero (29+ / 19-) | 0.050 | 0.251 | no |

Significant after Holm correction: **Adjacent Pair Accuracy**.

### LLM-Guided − Fixed Settings

| Metric | Mean Δ | 95% CI | Test | Discordant / non-zero pairs | p | p (Holm) | Significant |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Exact Match | +0.1200 | [+0.0500, +0.1900] | McNemar (exact) | 14 (A only 13, B only 1) | 0.002 | 0.002 | yes |
| Sequence Similarity | +0.0717 | [+0.0381, +0.1125] | Wilcoxon | 64 non-zero (45+ / 19-) | <0.001 | <0.001 | yes |
| Edit Similarity | +0.0929 | [+0.0587, +0.1340] | Wilcoxon | 78 non-zero (58+ / 20-) | <0.001 | <0.001 | yes |
| Adjacent Pair Accuracy | +0.1483 | [+0.1166, +0.1824] | Wilcoxon | 75 non-zero (69+ / 6-) | <0.001 | <0.001 | yes |
| Longest Correct Run | +0.1017 | [+0.0642, +0.1464] | Wilcoxon | 58 non-zero (49+ / 9-) | <0.001 | <0.001 | yes |
| Kendall Tau | +0.1828 | [+0.1284, +0.2472] | Wilcoxon | 77 non-zero (60+ / 17-) | <0.001 | <0.001 | yes |

Significant after Holm correction: **Exact Match, Sequence Similarity, Edit Similarity, Adjacent Pair Accuracy, Longest Correct Run, Kendall Tau**.

## E. Where the Bottleneck Is

### Junction scorer ranking (search-independent)

n/a - requires field: `junction_ranking` (per-sample top-1/top-3/MRR). The dense junction score matrix is not stored in samples.jsonl, and recomputing it needs pLM inference, which this offline rebuild does not do. Runs from the instrumentation change onward record it at zero extra model cost; for older runs, `python -m evaluation.junction_ranking` measures it separately.

### Selection signal trust

The run keeps whichever candidate scores best on the validity signal, so the signal's ability to rank candidates bounds what the search can deliver. **0.50 is a coin flip.**

| Measurement | Value |
| --- | --- |
| Samples with comparable candidate pairs | 84 |
| Comparable candidate pairs | 601 |
| Mean within-sample concordance | 0.7498 |
| Samples where concordance > 0.50 | 70 |
| Validity junction window | n/a |
| Validity confirmed penalty | n/a |

### Selection ceiling (Best Candidate)

| Metric | LLM-Guided | Best Candidate | Gap | Samples with a gap |
| --- | --- | --- | --- | --- |
| Exact Match | 0.2900 | 0.3700 | +0.0800 | 8/100 |
| Sequence Similarity | 0.5768 | 0.6441 | +0.0673 | 44/100 |
| Edit Similarity | 0.6622 | 0.7462 | +0.0840 | 49/100 |
| Adjacent Pair Accuracy | 0.7338 | 0.7826 | +0.0488 | 36/100 |
| Longest Correct Run | 0.5240 | 0.5821 | +0.0582 | 31/100 |
| Kendall Tau | 0.6242 | 0.7240 | +0.0997 | 54/100 |

On Adjacent Pair Accuracy the run leaves **0.0488** on the table in candidates it had already generated but did not select (36/100 samples). That is the size of the prize for a better selection signal alone.

### Trypsin filter recall

The filter pruned **7.63%** of candidate junctions on average. Whether any pruned junction was a *true* one is n/a - requires field: `trypsin_recall` (which junctions the trypsin filter pruned). Older runs stored only the pruned COUNT (`num_pruned`), which cannot tell us whether a pruned junction was a true one. Recorded from the instrumentation change onward.

## F. Difficulty Stratification and Error Modes

### Adjacent Pair Accuracy by fragment count

Difficulty scales with how many pieces the protein was cut into: the number of possible orderings grows factorially, while the pLM's evidence per junction does not improve. Lift over the Random Order floor is the honest read of whether the method is doing anything at each difficulty.

| Fragments | n | Random Order | LLM-Guided | Lift (paired) |
| --- | --- | --- | --- | --- |
| 2-4 | 6 | 0.6389 | 1.0000 | +0.3611 |
| 5-9 | 24 | 0.1780 | 0.8070 | +0.6291 |
| 10-19 | 37 | 0.0635 | 0.7370 | +0.6735 |
| 20-49 | 24 | 0.0531 | 0.6486 | +0.5955 |
| 50+ | 9 | 0.0292 | 0.5752 | +0.5459 |

### Every metric by fragment count

| Metric | 2-4 | 5-9 | 10-19 | 20-49 | 50+ |
| --- | --- | --- | --- | --- | --- |
| Exact Match | 1.0000 | 0.5417 | 0.2432 | 0.0417 | 0.0000 |
| Sequence Similarity | 1.0000 | 0.8952 | 0.5858 | 0.3058 | 0.1311 |
| Edit Similarity | 1.0000 | 0.7983 | 0.6725 | 0.5459 | 0.3416 |
| Adjacent Pair Accuracy | 1.0000 | 0.8070 | 0.7370 | 0.6486 | 0.5752 |
| Longest Correct Run | 1.0000 | 0.7557 | 0.5338 | 0.2973 | 0.1528 |
| Kendall Tau | 1.0000 | 0.7846 | 0.6489 | 0.4711 | 0.2532 |

### N-terminal start

| Measurement | Value |
| --- | --- |
| P(correct N-terminal start) | 0.9300 |
| ...on shuffled orderings | 0.1200 |
| Exact Match \| correct start | 0.3118 (29/93) |
| Exact Match \| wrong start | 0.0000 (0/7) |

Exact reconstruction is effectively conditional on getting the first fragment right — an ordering that starts wrong has already displaced every fragment after it.

### Breakpoints

| Measurement | Value |
| --- | --- |
| Samples | 100 |
| Mean breakpoints per protein | 6.25 |
| Median | 4.00 |
| Min / Max | 0 / 47 |
| Mean breakpoints per join | 0.2662 |
| Proteins assembled with 0 breakpoints | 29 |

### Error taxonomy

Failures are classified from the stored metric values (which were computed with the correct fragment-string semantics), checked in a fixed order so each sample lands in exactly one class. The cut points are disclosed in the table note and affect labelling only — no headline number depends on them.

| Failure mode | Samples | Share |
| --- | --- | --- |
| Exact reconstruction | 29 | 29.0% |
| Local transposition | 5 | 5.0% |
| Wrong start (structured, misanchored) | 6 | 6.0% |
| Partial assembly (correct start) | 60 | 60.0% |

## G. Cost

| Measurement | Value |
| --- | --- |
| LLM model | gpt-5-mini |
| Total LLM calls | 400 |
| Total LLM tokens | 604473 |
| LLM calls per sample | 4.00 |
| LLM tokens per sample | 6044.7 |
| Lever-choice failures | 0 |
| Wall clock per sample (total) | 49.9 s |
| Wall clock per sample (agentic arm) | 42.3 s |
| Wall clock per sample (control arm) | 7.5 s |
| LLM-Guided / Random Search time ratio | 5.62 |
| Completed samples | 100/100 |
| True order recovered | 100/100 |

The agentic arm costs **5.6x** the control arm's wall clock and **4.0 LLM calls per sample**, against which section D's paired tests are the return.

## Provenance

- Source: `samples.jsonl` in this folder — per-sample stored data only.
- No GPU, no model loading and no network access were used to build this report.
- Regenerate with `python -m evaluation.rebuild --run 140726_132208_agentic`.
- `report.md` (the run's original report) is left untouched; this file is additive.
