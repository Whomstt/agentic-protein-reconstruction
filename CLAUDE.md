# Agentic Protein Reconstruction

Reconstruct protein sequences from unordered trypsin-digestion fragments. An LLM
agent iterates over reconstruction hypotheses, scores each with an ESM-2 based
validity signal, and keeps the best candidate.

## Architecture

```
main.py
  ├── sweep.enabled: true  → evaluation/sweep.py runs each config.yaml sweep.grid
  │                          combo as its own main.py subprocess
  └── sweep.enabled: false → evaluation/runner.py::run_agentic() / run_sequential()
              ▼
      agents/iterative_runner.py::run_iterative_reconstruction
              ▼
      agents/react_agent.py::build_agent()      (LLM from llm_model.profile)
              │
              ├── run.calling_mode "single_call" (shipped): one LLM call per
              │   iteration returns the five levers; the harness runs the fixed
              │   pipeline in Python
              └── run.calling_mode "react": a LangGraph ReAct agent drives every
                  tool call itself (~4-6 LLM calls per iteration)
              │
              └── trypsin_filter → overlap_graph → junction_scorer
                                 → beam_search → validity_scorer
```

The agent runs up to `search.max_iterations` rounds per protein. Each round tries
a materially different strategy based on why the previous candidate scored
poorly, and the run keeps whichever candidate scores best on the validity signal.

`agents/react_agent.py::build_agent()` returns a mode-tagged `Agent` (`.graph`
for react, `.llm` for single_call) so the runner does not care which mode is on.
With `run.iteration1_deterministic: true` (shipped), iteration 1 runs the fixed
`search.default_levers` with no LLM call and becomes the report's Deterministic
arm. `trypsin_filter` and `overlap_graph` run once, on iteration 1. In
single_call mode, iterations 2+ skip the junction rescore when the LLM leaves
`junction_window` unchanged and reuse the cached `state["scores"]`.

## The five levers

Everything else (iteration budget, model choice, `max_length`, `batch_size`,
replica count) is fixed and off-limits to the agent.

| Lever | Where it applies |
| --- | --- |
| junction masking window | `junction_scorer(window=)` / `beam_search(window=)` |
| search mode | `beam_search(search_mode="beam"\|"greedy")` |
| beam width | `beam_search(beam_width=)` |
| edge mode | `beam_search(edge_mode="hard"\|"soft")` |
| confirmed-edge bonus | `beam_search(confirmed_bonus=)` |

## Evaluation arms

`run_agentic` produces up to four paired arms per protein plus an oracle ceiling,
all on the same protein:

- **Shuffled baseline** — a random fragment ordering. A floor, not a method.
- **Deterministic** — iteration 1 with fixed `default_levers`, no LLM.
- **Control** (`run.control_baseline.enabled`) — same budget, pipeline and
  best-validity selection as the agent, but levers come from a non-LLM policy
  (`agents/baseline_policy.LeverPolicy`, `random` or `grid`) instead of the LLM.
  Costs 0 LLM calls. `Δ Agentic − Control` isolates the value of the LLM's
  reasoning from the value of trying several candidates and keeping the best.
- **Agentic** — the LLM-driven arm; the kept candidate is the best-validity one
  across all iterations, subject to `search.improvement_margin`.
- **Oracle ceiling** (`run.report_oracle`) — per metric, the best true value
  among the candidates the agent actually generated. It peeks at ground truth, so
  it is a ceiling, not a method; the Oracle − Agentic gap is what imperfect
  selection leaves on the table.

Iteration 1 is a shared starting point for the agentic and control arms, so the
agent can never score worse than the Deterministic arm on validity. The
meaningful comparisons are therefore the true-metric columns, and
`Δ Agentic − Control` in particular.

## System layers

- `tools/` — LangChain `@tool` wrappers over the algorithms, storing run-time
  artifacts in `tools/state.py`. Their docstrings are the tool descriptions the
  LLM reads in react mode, so they document the diagnostics deliberately.
- `algorithms/` — pure computation, no LangChain dependency.
- `models/` — HuggingFace loaders for ESM-2 / ProtBERT. Junction and validity
  scoring share model instances and reset the ESM rotary cache between calls.
- `agents/iterative_runner.py` — the multi-iteration loop, per-iteration history,
  best-candidate selection.
- `agents/deterministic_agent.py` — single_call lever selection (`LeverChoice`)
  and the shared tool pipeline (`_score_lever_values`), reused by the LLM arm and
  the control arm.
- `evaluation/runner.py` — the single sample loop behind `run_sequential` and
  `run_agentic`.
- `evaluation/sweep.py` / `sweep_report.py` — grid orchestration and the combined
  cross-combo report.

Shared state keys: `fragment_samples`, `fragments`, `iteration_history`,
`best_iteration`, `best_reconstruction`, `best_validity_score`, `best_order`,
`search_strategy`, `reconstruction`, `order`, `validity_score`.

## Tools

1. `trypsin_filter(fragments)` — trypsin constraints and beam hints. Sets
   `impossible_junctions`, `missed_cleavage_fragments`, `start_candidates`.
2. `overlap_graph(fragment_samples)` — hard adjacency edges from multi-replica
   digestions. Sets `confirmed_junctions`, `confirmed_adjacencies`,
   `confirmed_successors`, `unscored_junctions`.
3. `junction_scorer(window, junction_pairs)` — masked-LM scores for ordered
   fragment pairs; can rescore a targeted subset and merge it into the matrix.
   Feedback: mean/min/max score and count. A narrow spread means this window is
   not discriminating real junctions from wrong ones.
4. `beam_search(search_mode, beam_width, edge_mode, confirmed_bonus, window)` —
   the ordering search. Beam falls back to greedy extension if constraints cut it
   off; greedy falls back to the least-bad candidate when every option is
   trypsin-impossible. Feedback: `fell_back`, `forced_impossible_count`,
   confirmed edges realized/total, `mean_junction_score`.
5. `validity_scorer(reconstruction)` — the selection signal, returned as a
   breakdown (`validity_score`, `junction_local_ppl`,
   `confirmed_adjacency_agreement`, `confirmed_penalty_applied`) so the next
   lever choice can target the weak point. Falls back to whole-sequence
   pseudo-perplexity if handed a string whose ordering is unknown.

In react mode the LLM reads these dicts directly; in single_call mode the harness
threads the previous iteration's breakdown, junction stats and beam diagnostics
into the next prompt.

## Validity score (selection signal)

Candidates reuse the same fragments and differ only at the junctions, so the
signal scores only what varies:

```
junction_local_ppl * (1 + confirmed_penalty * (1 - confirmed_adjacency_agreement))
```

- `junction_local_ppl` = `exp(-mean_junction_logprob)` over non-confirmed
  boundaries. Whole-sequence pseudo-perplexity is ~95% invariant across orderings
  and drowns the signal.
- `confirmed_adjacency_agreement` = fraction of the overlap graph's confirmed
  directed edges placed consecutively; a near-ground-truth structural signal that
  strengthens with replica count.

`trypsin_filter`'s impossible junctions are excluded by construction in
`beam_order`/`greedy_order`, and confirmed junctions are excluded from the
PLM-scored set and covered by the agreement term instead.

At window 5 and penalty 0.75, concordance with true quality is ~57% on yeast and
~61% on E. coli. This is a plausibility score, not an oracle: a candidate can
score well and still be ordered wrong.

## Metrics

The task is a permutation of a fixed fragment set, so composition is invariant
and only ordering varies.

- **exact_match** — `target == reconstruction`.
- **similarity** — `difflib.SequenceMatcher` ratio, the one soft string metric.
- **adjacent_pair_acc** — fraction of true adjacent pairs preserved. Directed and
  string-multiset based. Primary ordering metric.
- **longest_correct_run** — longest contiguous correct block / n.
- **kendall_tau** — global ordering correlation (0 random, 1 perfect, −1
  reversed).

No metric rewards a reversal. Ground-truth order is not stored;
`recover_true_order` re-derives it by longest-first tiling (replica 0 tiles the
target by construction). When tiling fails the three ordering metrics are NaN,
`true_order_recovered` is false, and those samples drop out of the averages via
`nanmean`. Two pipeline assumptions are measured rather than assumed: junction
ranking (`python -m evaluation.junction_ranking`, search-independent
top-1/top-3/MRR) and validity-signal trust (the Validity Signal Concordance
report section).

## Configuration

All access goes through `from config import cfg`; the file is
[config.yaml](config.yaml).

- `run.method` — `agentic` or `sequential`, read when `sweep.enabled` is false.
- `run.calling_mode` — `single_call` (shipped) or `react`.
- `run.iteration1_deterministic` — `true` (shipped) makes iteration 1 the no-LLM
  Deterministic arm; `false` makes it a genuine LLM lever choice. Only affects
  single_call mode. Labels follow automatically via
  `evaluation/reporting.py::first_pass_label()`. Disclose it in any writeup.
- `run.control_baseline` — the matched-budget non-LLM control arm. Keep
  `lever_space` matched to the agent's plausible range so the comparison is fair.
- `run.report_oracle` — adds the Oracle column and Selection Ceiling section, free.
- `search.max_iterations` / `early_stop_patience` — equal values (5 and 5 shipped)
  mean a fixed budget with no early stopping. The lever space is small, so returns
  diminish after a handful of genuinely distinct attempts.
- `search.improvement_margin` — relative validity drop a later iteration must
  clear to replace the incumbent; `0.0` (shipped) is a raw argmin. A positive
  value is a winner's-curse guard that matters more at higher budgets, since the
  validity signal is only ~57-61% concordant.
- `search.default_levers` — the single source of truth for all five levers, used
  for iteration 1, as the react-mode fallback, and as the pipeline-wide default
  read by `algorithms/` and `tools/` (including `run_sequential`, which has no
  agent at all).
- `search.validity_junction_window` / `validity_confirmed_penalty` — the
  selection signal's two constants.
- `llm_model.sampling.*` — resolved by `react_agent._llm_sampling_kwargs()`; only
  non-null keys are sent. The shipped `gpt-5*` deployments reject non-default
  `temperature`/`top_p`, so steering is via `reasoning_effort` and `verbosity`.
- `sweep.grid` — axes recognized by `evaluation/sweep.py::_apply_overrides`:
  `organism`, `replica_count`, `mlm_profile`, `method`,
  `iteration1_deterministic`, `max_iterations`, `early_stop_patience`,
  `improvement_margin`, `test_samples`. A swept `max_iterations` pins
  `early_stop_patience` to it unless that is swept too.

**Leakage note.** Nothing is trained, and there is no train/test split. Constants
chosen offline (`improvement_margin`, `validity_junction_window`,
`validity_confirmed_penalty`) came from the same undivided per-organism pool that
evaluation samples are drawn from, so they are not disjoint from what is
reported. Treat them as disclosed sensitivity choices.

## Data

JSONL records with `fragments`, an organism-specific original (`ecoli_original` /
`yeast_original`), `target_reconstruction`, `num_fragments`, `replica_count`,
`missed_cleavage_ratio`.

`preprocessing/preprocessing.py` filters the reviewed UniProt FASTA to the active
organism, deduplicates by gene (`GN=`), and generates `replica_count` digestion
replicas per protein. Each output has a `.meta.json` sidecar;
`ensure_fresh_dataset()` compares it against the active config and regenerates
when organism, replica count or missed-cleavage ratio changed, and `main.py`
calls it before every non-sweep run. Only `trypsin_digest` is wired in; the other
enzymes are unused extension points. `evaluation/runner.py::_load_test_samples`
shuffles the pool with `misc.seed` and takes the first `data.test_samples`
records.

## Evaluation and reporting

Each run writes `results/<timestamp>_<name>/` with `summary.json`,
`samples.jsonl` (full per-sample and per-iteration history), `report.md` and
`metric_comparison.svg`.

Every run then also produces a statistical layer, generated by
`evaluation/runner.py::_build_analysis_artifacts()` and re-runnable offline over
any past run:

```bash
python -m evaluation.rebuild --all            # every run + cross-run report
python -m evaluation.rebuild --run <folder>   # one run
```

No GPU, no model loading, no network: everything derives from `samples.jsonl`
plus the config snapshot. Per run: `analysis_report.md` (sections A-G),
`results.csv`, `summary.csv`, `tables/*.tex`, `figures/*.pdf` (+ PNG twins).
Cross-run output lands in `results/_analysis/`. The run's original `report.md` is
never overwritten, and the hook is wrapped so a reporting bug cannot fail a
finished run.

- `evaluation/stats.py` — Wilson intervals (Exact Match is a binomial count), BCa
  bootstrap CIs (fixed seed), exact McNemar and Wilcoxon signed-rank for the
  paired arms, Holm correction across the five metrics. stdlib only.
- `evaluation/analysis.py` — per-sample derivations: fragment bins, breakpoints,
  N-terminal start, error taxonomy, concordance, oracle gap.
- `evaluation/exports.py` / `figures.py` — CSV and booktabs LaTeX from shared row
  data, and the figure house style.
- `evaluation/thesis_tables.py` — the report's Results tables
  (`python -m evaluation.thesis_tables --run <folder>` → `report/tables/*.tex`).
  A composition layer: intervals and paired tests are imported from
  `evaluation/rebuild.py`, aggregations from `analysis.py`, rendering from
  `exports.py`, so these tables and `analysis_report.md` cannot disagree. Files
  are written camera-ready (no provenance comments) because they are `\input{}`
  into the paper. Computed nowhere else: the agent-behaviour table (which levers
  the LLM moved, and how often a later iteration displaced iteration 1) and the
  experimental-configuration table, derived from every run's config snapshot.
  `tests/test_thesis_tables.py` re-derives values from `samples.jsonl` with no
  `evaluation/` imports and asserts them against the emitted cells.

Tables written by `rebuild.py` under `results/*/tables/` do carry a provenance
comment (source run, command, row count, timestamp) via `exports.stamp_tables`.

An agentic `report.md` leads with **Shuffled → Deterministic → Agentic Best** plus
the paired per-sample gain distribution, adding the Control arm and
`Δ Agentic − Control` when that arm ran, and the Oracle column plus Selection
Ceiling section when enabled. Two sections are always emitted: Validity Signal
Concordance and Cost, Efficiency & Completion. For a significance claim, use the
paired Wilcoxon on per-sample gains — `Agentic − Control` for the reasoning
claim, `Agentic − Deterministic` for the end-to-end gain.

## Research validity notes

- **No ground-truth leakage.** `target_reconstruction` is read only in
  `run_agentic()` after `run_iterative_reconstruction()` returns, purely for
  scoring. It never reaches the agents or any tool; the LLM never sees the true
  sequence.
- **No hidden clamping or best-of-N.** The levers the LLM returns pass straight
  to the tools with no post-hoc override, and there is no silent retry or
  resample that keeps only a favourable call.
- **Disclosure items.** `run.iteration1_deterministic`,
  `search.improvement_margin`, and the leakage note above.
- **Prompt autonomy.** The lever prompts give a mechanistic description of each
  lever and diagnostic plus directional guidance ("narrower window for local
  motifs, wider for more context"); the agent picks every actual value. The
  diagnostic-interpretation guidance is the one piece of prompt design to
  disclose.

## Commands

```bash
python main.py                        # single entry point, driven by config.yaml
python -m preprocessing.preprocessing # force a dataset rebuild
python -m evaluation.sequential       # deterministic baseline only
python -m evaluation.agentic          # agentic evaluation only
python -m evaluation.junction_ranking # search-independent junction-ranking check
python -m evaluation.rebuild --all    # regenerate reports/CSVs/tables/figures
python -m evaluation.thesis_tables --run <folder>   # report/tables/*.tex

python tests/test_stats.py            # stdlib unittest; pytest is not a dependency
python tests/test_analysis.py
python tests/test_instrumentation.py
python tests/test_thesis_tables.py
```

## Conventions

- No `__init__.py` files; imports are flat and the repo is installed editable.
- All config access goes through `from config import cfg`.
- Algorithms stay pure; tools manage state.
- ESM-2 is the default MLM; ProtBERT is supported.
- Each model module exposes a module-level `model_lock`, held around every model
  call so junction and validity scoring can share the process without racing.
