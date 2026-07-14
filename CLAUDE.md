# Agentic Protein Reconstruction

Reconstruct protein sequences from unordered digestion fragments. An LLM agent iterates over multiple reconstruction hypotheses, scores each with an ESM-2 based validity model, and keeps the best candidate.

## Architecture

```
main.py
      │
      ├── sweep.enabled: true  → evaluation/sweep.py loops config.yaml's sweep.grid,
      │                          each combo re-running main.py as a subprocess
      │
      └── sweep.enabled: false → evaluation/runner.py::run_agentic() or run_sequential()
                  │              (picked by run.method)
                  ▼
      agents/iterative_runner.py::run_iterative_reconstruction
                  │
                  ▼
      agents/react_agent.py::build_agent()  (LLM from llm_model.profile)
                  │
                  ├── run.calling_mode: "single_call"  (shipped)
                  │     1 LLM call/iteration picks the 5 lever values (structured
                  │     output); the harness runs the fixed tool pipeline in Python
                  │
                  └── run.calling_mode: "react"
                        LangGraph ReAct agent drives every tool call itself
                        (~4-6 LLM calls/iteration)
                  │
                  ├── trypsin_filter
                  ├── overlap_graph
                  ├── junction_scorer
                  ├── beam_search
                  └── validity_scorer
```

The agent runs up to `search.max_iterations` rounds per sample. Each round tries a materially different strategy based on why the previous candidate scored poorly, and the run keeps whichever candidate scores best on the validity signal.

`run.calling_mode` controls how much the LLM drives. `"single_call"` (shipped) asks the LLM once per iteration for the five lever values via `agents/deterministic_agent.py`'s `LeverChoice` structured output; the harness (`agents/iterative_runner.py::run_iterative_reconstruction`) then runs junction scoring → beam/greedy search → validity scoring directly in Python. `"react"` instead lets a LangGraph `create_react_agent` decide every tool call itself, costing several LLM calls per iteration. `agents/react_agent.py::build_agent()` returns a mode-tagged `Agent` (`.graph` for react, `.llm` for single_call) so `evaluation/runner.py` does not need to know which mode is active.

With `run.iteration1_deterministic: true` (shipped), iteration 1 runs the fixed `search.default_levers` with no LLM call and the agent refines from it in iterations 2+; this makes iteration 1 the report's Deterministic arm. With `false`, iteration 1 is itself an LLM lever choice. `trypsin_filter` and `overlap_graph` run once on iteration 1 in either mode. In `single_call` mode, iterations 2+ skip the `junction_scorer` rescore when the LLM keeps `junction_window` unchanged, reusing the cached `state["scores"]` and letting `beam_search`'s lazy-rescore check confirm no recompute is needed.

The agent's control surface is exactly five levers:

- junction masking window — `junction_scorer(window=...)` / `beam_search(window=...)`
- search mode — `beam_search(search_mode="beam"|"greedy")`
- beam width — `beam_search(beam_width=...)`
- edge mode — `beam_search(edge_mode="hard"|"soft")`
- confirmed-edge bonus — `beam_search(confirmed_bonus=...)`

Everything else — iteration budget, PLM/model choice, `max_length`, `batch_size`, replica count — is fixed and off-limits to the agent.

## Evaluation Arms

`run_agentic` produces up to four paired arms per sample plus an oracle ceiling, all on the same protein:

- **Shuffled baseline** — a random fragment ordering (the floor, not a method).
- **Deterministic** — iteration 1 with fixed `default_levers`, no LLM (present when `run.iteration1_deterministic: true`).
- **Control** (`run.control_baseline.enabled: true`) — the same iteration budget, tool pipeline and best-validity selection as the agent, but the five levers are chosen by a non-LLM policy (`agents/baseline_policy.LeverPolicy`, `policy: "random"` seeded-uniform over `lever_space` or `"grid"` = a fixed spread across the grid) instead of the LLM. Runs paired and costs 0 LLM calls. `Δ Agentic − Control` isolates the value of the LLM's reasoning from the value of trying several diverse candidates and keeping the best. Driven via `run_iterative_reconstruction(control_policy=...)` → `agents/deterministic_agent.run_policy_iteration()`; always a policy→fixed-pipeline loop regardless of `run.calling_mode`.
- **Agentic** — the LLM-driven arm; the kept candidate is the best-validity one across all iterations (subject to `search.improvement_margin`).
- **Oracle ceiling** (`run.report_oracle: true`) — for each metric, the best true-metric value among the candidates the agent actually generated. It peeks at ground truth, so it is a ceiling, not a method; the Oracle − Agentic gap is what the imperfect validity selection leaves on the table.

`run.iteration1_deterministic` is honoured identically in the agentic and control arms, so iteration 1 is a shared starting point and only iterations 2+ differ by lever source. Because iteration 1 is inside the agent's candidate set, the agent can never score worse than the Deterministic arm on validity — so the meaningful comparisons are the true-metric columns, and `Δ Agentic − Control` in particular.

## System Layers

- `tools/` — LangChain `@tool` wrappers, thin adapters around the algorithms that store run-time artifacts in `tools/state.py`.
- `algorithms/` — Pure computation, no LangChain dependencies.
- `models/` — HuggingFace loaders for ESM-2 or ProtBERT. Junction and validity scoring share model instances and reset the ESM rotary cache between calls.
- `agents/iterative_runner.py` — Orchestrates the multi-iteration loop, prompts the LLM with prior scores/strategy/history, stores per-iteration history, and applies best-candidate selection.
- `agents/deterministic_agent.py` — `single_call` lever selection (`LeverChoice`) and the shared tool pipeline (`_score_lever_values`), reused by both the LLM arm (`run_single_call_iteration`) and the control arm (`run_policy_iteration`).
- `agents/baseline_policy.py` — `LeverPolicy`, the non-LLM lever selector for the control arm.
- `evaluation/runner.py` — Single sample loop for both `run_sequential` and `run_agentic`; `main.py`, `evaluation/sequential.py`, and `evaluation/agentic.py` all call into it instead of duplicating the loop.
- `evaluation/sweep.py` / `evaluation/sweep_report.py` — grid sweep orchestration and the combined cross-combo markdown report.

## Iterative Agent Loop

Each iteration's prompt tells the LLM to explain why the previous attempt likely failed, pick a materially different tactic using only the five levers, rescore only the junction pairs that need fresh scores, produce a fresh candidate, and score it with `validity_scorer`. The run stops early once the best validity score fails to improve for `search.early_stop_patience` consecutive iterations, otherwise it runs the full `search.max_iterations`. Both agent modes inject the entire iteration history as a per-attempt `levers -> validity` digest plus an explicit "best so far" target, so the agent knows which combinations it has already tried and the incumbent score it must beat. Per-iteration records store `lever_values` and `changed_levers` for auditability.

Best-candidate selection is controlled by `search.improvement_margin`. The incumbent (starting from iteration 1) is replaced only when a later iteration's validity score is lower by more than the relative margin; `0.0` (shipped) makes selection a raw argmin — the best-scoring candidate wins outright. A positive margin is a winner's-curse guard: the validity signal is imperfect (~57–61% concordant with true quality), so a raw argmin over many diverse candidates can occasionally pick a noise-lower-but-worse ordering. That risk grows with `max_iterations`, so if you push the budget toward ~10 a small positive margin (~0.03) is worth restoring. Any such value is a disclosed sensitivity choice tuned on non-disjoint data (see the leakage note under Configuration), not a validated constant.

Shared state keys: `fragment_samples`, `fragments`, `iteration_history`, `best_iteration`, `best_reconstruction`, `best_validity_score`, `best_order`, `search_strategy`, `reconstruction`, `order`, `validity_score`.

## Tools

1. `trypsin_filter(fragments)`

   Initializes trypsin-derived constraints and beam hints. State keys: `impossible_junctions`, `missed_cleavage_fragments`, `start_candidates`.

2. `overlap_graph(fragment_samples)`

   Builds hard adjacency edges from multi-sample digestions. State keys: `confirmed_junctions`, `confirmed_adjacencies`, `confirmed_successors`, `unscored_junctions`.

3. `junction_scorer(window=None, junction_pairs=None)`

   Scores ordered fragment pairs with a masked language model. The first `W` residues of the successor fragment are masked one at a time and averaged, where `W` defaults to `search.default_levers.junction_window`. Accepts a targeted subset of junction pairs to rescore and merges the results into the existing score matrix.

   **Feedback signal:** `mean_score`/`min_score`/`max_score`/`num_junctions_scored` over the pairs just scored. A narrow spread near the mean means the current window is not giving the PLM enough signal to discriminate real junctions from wrong ones — try a different window rather than proceeding with weak scores.

4. `beam_search(search_mode="beam", beam_width=None, edge_mode="hard", confirmed_bonus=0.0, window=None)`

   Reconstructs an ordering from the score matrix. Beam mode falls back to greedy extension if constraints cut the search off before all fragments are placed; greedy falls back to the least-bad remaining candidate when every option at a step is trypsin-impossible.

   **Feedback signal:** `fell_back` (beam mode: `beam_width`/`edge_mode` cut the search short — widen the beam or try `edge_mode="soft"`), `forced_impossible_count` (greedy mode: trypsin constraints fought the ordering — try `search_mode="beam"`), `num_confirmed_edges_realized`/`num_confirmed_edges_total`, and `mean_junction_score` (a cheap ordering-quality preview before paying for `validity_scorer`).

5. `validity_scorer(reconstruction=None)`

   Scores the ordering currently in shared state as the basis for best-candidate selection, combining junction-local pseudo-perplexity over non-confirmed junctions with confirmed-adjacency agreement (see the Validity Score section). If an explicit reconstruction with unknown ordering is passed, it falls back to whole-sequence pseudo-perplexity on that string.

   **Feedback signal:** a dict, not a bare float — `validity_score` (lower is better; this decides which candidate wins) plus its two components `junction_local_ppl` and `confirmed_adjacency_agreement`, and `confirmed_penalty_applied`. Exposing the components lets the next lever choice target the weak point: high `junction_local_ppl` → change `junction_window`; low `confirmed_adjacency_agreement` → switch `edge_mode` to `"hard"` or raise `confirmed_bonus` in `"soft"` mode.

   **Propagation across modes:** in `react` mode the LLM reads these dicts directly in tool messages. In `single_call` mode the harness threads the previous iteration's `validity_breakdown`, `junction_stats`, and `beam_diagnostics` into the next iteration's prompt text so the LLM knows not just the previous score but why it scored that way.

## Metrics

The task is a permutation of a fixed fragment set: string composition is invariant, so the only thing that varies is ordering. Five quality metrics:

- **exact_match** — `target == reconstruction`. A sanity floor, rare on this task.
- **similarity** — `difflib.SequenceMatcher` ratio; the single soft string metric.
- **adjacent_pair_acc** — fraction of true adjacent fragment pairs preserved. Directed and string-multiset based, so duplicate/substring-identical fragments do not read as misordered. Primary ordering metric.
- **longest_correct_run** — longest contiguous correctly-ordered fragment block / n; credits partial assembly.
- **kendall_tau** — global ordering correlation (0 = random, 1 = perfect, −1 = reversed). Ranks are matched on fragment strings (`_matched_rank_sequence`) so duplicates do not corrupt it.

No metric rewards a reversed sequence (kendall → −1, adjacent_pair_acc → 0 on a clean reversal).

Ground-truth order is not stored. `recover_true_order` re-derives it by greedy longest-first tiling of the target with the fragments (replica 0 tiles the target by construction). When tiling fails, the three ordering metrics are `NaN` (not a misleading `0.0`), `compute_all` sets `true_order_recovered=False`, and those samples are dropped from the averages by `nanmean`; the usable-sample count surfaces in the Cost/Completion section. `is_clean_permutation` validates predictions for the completion count.

Two pipeline assumptions are measured, not taken on faith:

- **Junction-scorer ranking**, search-independent — `python -m evaluation.junction_ranking` scores a dense junction matrix over all ordered pairs and reports top-1/top-3 successor accuracy and MRR of the true successor.
- **Validity-signal trust** — the in-report Validity Signal Concordance section measures whether lower validity actually corresponds to higher true quality among the candidates each sample tried. A pLM plausibility score rewards natural-looking proteins, a poor prior for novel targets, so this concordance is the number to watch, not the validity value itself.

## Validity Score (selection signal)

Candidate orderings reuse the same fragments and differ only at the junctions, so the selection signal scores only what varies:

```
junction_local_ppl * (1 + confirmed_penalty * (1 - confirmed_adjacency_agreement))
```

- **junction_local_ppl** = `exp(-mean_junction_logprob)` over the ordering's non-confirmed boundaries only. Whole-sequence pseudo-perplexity is ~95% invariant across orderings (the within-fragment residues are identical), which drowns the ordering signal; scoring only the junctions keeps the residues whose likelihood actually depends on the order.
- **confirmed_adjacency_agreement** = fraction of the overlap graph's confirmed directed edges realized as consecutive in the ordering — a near-ground-truth structural signal (real multi-replica overlaps) that strengthens with replica count.

The two hard-constraint tools feed this signal rather than being re-scored themselves: `trypsin_filter`'s `impossible_junctions` are excluded from every ordering by construction in `beam_order`/`greedy_order`, so they need no validity check; `overlap_graph`'s `confirmed_junctions` are excluded from the PLM-scored junction set in `junction_local_ppl` and scored via `confirmed_adjacency_agreement` instead.

At junction window 5 and confirmed penalty 0.75, concordance with true reconstruction quality is ~57% on yeast and ~61% on E. coli. This is a plausibility/consistency score, not an exact-match oracle: a candidate can score well and still be wrong in sequence order.

## Configuration

Config lives in [config.yaml](config.yaml); all access goes through `from config import cfg`.

- `run.method` — `"agentic"` (LLM-driven) or `"sequential"` (deterministic, no LLM); read when `sweep.enabled` is false.
- `run.calling_mode` — `"single_call"` (shipped) or `"react"` (see Architecture).
- `run.iteration1_deterministic` — `true` (shipped): iteration 1 is the fixed-`default_levers` Deterministic arm with no LLM call, and the agent refines from it. `false`: iteration 1 is itself a genuine `LeverChoice` LLM decision and no deterministic arm is produced inside the run. Only affects `single_call` mode (react always drives iteration 1 through the LLM). Labels adjust automatically via `evaluation/reporting.py::first_pass_label()` / `run_type_summary()`. Requires explicit disclosure in any writeup.
- `run.control_baseline` — the matched-budget non-LLM control arm (see Evaluation Arms). `enabled: true` (shipped); `policy: "random"` or `"grid"`; `lever_space` is the value set the policy draws from — keep it matched to the agent's plausible range so the comparison is fair. `Δ Agentic − Control`, not `Δ Agentic − Deterministic`, is the defensible "does the LLM's reasoning help" number.
- `run.report_oracle` — `true` (shipped) adds the Oracle ceiling column and Selection Ceiling section. Free (no extra model calls).
- `search.max_iterations` — agentic refinement rounds. With `early_stop_patience` equal to it, this is a fixed per-sample budget with no early stopping (shipped: 5 and 5). The lever space is small (five levers, two of them binary), so returns diminish after a handful of genuinely-distinct attempts; ~5 is a reasonable balance and higher budgets mostly add cost and winner's-curse exposure.
- `search.early_stop_patience` — consecutive non-improving iterations before stopping early; equal to `max_iterations` disables early stopping so every sample runs the full budget.
- `search.improvement_margin` — relative validity drop a later iteration must clear to replace the incumbent best; `0.0` (shipped) is raw argmin. A winner's-curse guard whose value matters more at higher `max_iterations` (see the Iterative Agent Loop section).
- `search.default_levers` — the single source of truth for all five levers (`junction_window`, `search_mode`, `beam_width`, `edge_mode`, `confirmed_bonus`) across the whole pipeline. Iteration 1 uses these directly when `run.iteration1_deterministic`; in react mode they are the fallback when the LLM omits a lever argument; and `junction_window`/`beam_width` are the default read directly by `algorithms/score_junctions.py`, `algorithms/beam_order.py`, `tools/junction_scorer.py`, and `tools/beam_search.py` — including `run_sequential`, which has no agent at all.
- `search.validity_junction_window` — residues of each junction scored for the selection signal (default 5).
- `search.validity_confirmed_penalty` — weight on `(1 - confirmed_adjacency_agreement)` in the selection signal (default 0.75).
- `validity_model.*` — model settings used only for the whole-sequence pseudo-perplexity fallback.
- `llm_model.sampling.*` — resolved by `agents/react_agent.py::_llm_sampling_kwargs()`. Only non-null keys are passed to `ChatOpenAI`, so a null means "use the model's default" (and, for an unsupported knob, "never send it"). The shipped `gpt-5*` deployments are reasoning models that reject any `temperature`/`top_p` other than the default, so both stay null and the model is steered with `reasoning_effort` (`minimal`/`low`/`medium`/`high`) and `verbosity` (`low`/`medium`/`high`); higher `reasoning_effort` trades latency and tokens for deeper reasoning. `seed` gives best-effort determinism of the lever choices. `top_k` is not an OpenAI parameter and is forwarded (via `model_kwargs`) only to non-OpenAI backends.
- `sweep.grid` — axes to sweep; each combo gets its own `results/` folder plus one combined cross-combo report. Recognized axis keys (`evaluation/sweep.py::_apply_overrides`): `organism`, `replica_count`, `mlm_profile`, `method`, `iteration1_deterministic`, `max_iterations`, `early_stop_patience`, `improvement_margin`, plus per-combo `test_samples`. A `max_iterations` axis auto-pins `early_stop_patience` to each swept value unless it is also swept. Unrecognized keys are ignored.

**Leakage note.** There is no train/test split anywhere in the project — no training happens. Any constant chosen offline (`improvement_margin`, `validity_junction_window`, `validity_confirmed_penalty`) was picked on samples drawn from the same undivided per-organism pool that `data.test_samples`/`sweep.test_samples` also draw evaluation samples from, so it is not disjoint from what gets reported. Treat these as disclosed sensitivity choices; a properly disjoint tune/eval split would need a held-out partition of the fragmented pool (e.g. by protein).

## Data

JSONL records with fields like `fragments`, organism-specific originals (`ecoli_original`, `yeast_original`), `target_reconstruction`, `num_fragments`, `replica_count`, `missed_cleavage_ratio`.

`preprocessing/preprocessing.py` filters the raw UniProt FASTA to the active organism, deduplicates by gene (`GN=` tag) so the same protein is not repeated across near-identical strains, then generates `replica_count` digestion replicas per protein. There is no train/test split, so preprocessing writes one deduped fragmented file per organism (`data.fragmented_ecoli`, etc.) and evaluation draws directly from it. Only `trypsin_digest` is wired into the pipeline; `lys_c_digest`/`asp_n_digest`/`glu_c_digest` are unused extension points for other cleavage enzymes.

Each fragmented output has a `.meta.json` sidecar recording the `organism`/`replica_count`/`missed_cleavage_ratio` it was generated with. `ensure_fresh_dataset()` compares that against the active config and regenerates the dataset if any of the three changed; `main.py` calls it before every non-sweep run, so editing `data.replica_count` (or organism, or missed-cleavage ratio) and running `python main.py` picks it up without a manual preprocessing step. Each sweep combo re-enters `main.py` as its own subprocess, so this check re-runs preprocessing once per distinct combination, not on every combo.

`evaluation/runner.py::_load_test_samples` shuffles the pool with the global `misc.seed` and takes the first `data.test_samples` records (the whole pool if `test_samples` is unset). `target_reconstruction` (or the organism-specific original) is the ground-truth target.

## Evaluation

- `evaluation/runner.py` — shared sample loop for `run_sequential` (deterministic baseline) and `run_agentic` (iterative agent + control/oracle arms).
- `evaluation/sequential.py` / `evaluation/agentic.py` — thin CLI wrappers, independent of `run.method`.
- `evaluation/metrics.py` — the five quality metrics plus the search-independent `junction_ranking_stats`, the selection-signal trust check `rank_concordance`, the `is_clean_permutation` completion guard, and `nanmean` for NaN-safe averaging.
- `evaluation/junction_ranking.py` — standalone diagnostic (`python -m evaluation.junction_ranking`) measuring how well the raw pLM junction scorer ranks true successor fragments, independent of search.
- `evaluation/reporting.py` — config snapshot, per-sample results, distribution stats, the markdown report, and `metric_comparison.svg`.
- `evaluation/sweep.py` / `evaluation/sweep_report.py` — grid sweep orchestration and the combined cross-combo report.

Each run's `results/<timestamp>_<name>/` folder contains `summary.json`, `samples.jsonl` (full per-sample and per-iteration history for auditability), `report.md`, and `metric_comparison.svg`.

An agentic report leads with the benchmark table — **Shuffled Baseline → Deterministic → Agentic Best**, with a `Δ Agentic − Deterministic` column — plus the paired per-sample gain distribution. When `run.control_baseline.enabled` is on the table also carries the Control arm and a `Δ Agentic − Control` column, with a dedicated "Agentic vs. Control (paired)" section; that Δ is the defensible reasoning claim. When `run.report_oracle` is on the table gains an Oracle column and a "Selection Ceiling (Oracle)" section. Columns and sections are emitted only for arms that were actually produced, so a run with both flags off renders the plain three-arm table. Two sections are always emitted for agentic runs: **Validity Signal Concordance** (does the selection signal track true quality, within-sample across iterations via `rank_concordance`; ~0.50 = chance) and **Cost, Efficiency & Completion** (LLM calls/tokens/wall-clock per sample, completion and lever-choice-failure rates, and agentic vs. control wall-clock). A "How to Read This Report" block (`run_type_summary`) names each column for the active config. The multi-arm table and its sections are built by `evaluation/reporting.py::_format_multi_arm_benchmark()` / `_format_agent_vs_control_distribution_rows()` / `_format_oracle_gap_rows()`.

For a significance claim, run a paired Wilcoxon signed-rank test on the per-sample gains — `Agentic − Control` for the reasoning claim, `Agentic − Deterministic` for the end-to-end gain.

## Research Validity Notes

- **No ground-truth leakage.** `target_reconstruction` is read only in `evaluation/runner.py::run_agentic()` after `run_iterative_reconstruction()` returns, purely for scoring; it never reaches `agents/iterative_runner.py`, `agents/deterministic_agent.py`, `agents/react_agent.py`, or any tool. The LLM never sees the true sequence.
- **No hidden clamping or best-of-N.** The five levers the LLM (or, in `single_call` mode, the `LeverChoice` structured output) returns pass straight to the tools with no post-hoc override, and there is no silent retry or resample that keeps only a favorable call.
- **Disclosure items.** `run.iteration1_deterministic`, `search.improvement_margin`, and the leakage note under Configuration are the settings to state explicitly in any writeup.
- **Prompt autonomy.** The lever-choice prompts do not seed the agent with human-chosen values. They provide a mechanistic description of what each lever does and what each diagnostic signal means, plus directional guidance (e.g. "narrower window for local motifs, wider for more context"); the agent decides every actual value itself. The diagnostic-interpretation guidance is the one piece of prompt design to disclose.

## Commands

```bash
python main.py                        # single entry point; behaviour controlled by config.yaml
python -m preprocessing.preprocessing # regenerate the fragmented dataset for the active organism
python -m evaluation.sequential       # deterministic baseline only, bypassing run.method
python -m evaluation.agentic          # agentic evaluation only, bypassing run.method
python -m evaluation.junction_ranking # search-independent pLM junction-ranking diagnostic (top-1/top-3/MRR)
```

## Conventions

- No `__init__.py` files; imports are flat.
- All config access goes through `from config import cfg`.
- Algorithms stay pure; tools manage state.
- ESM-2 is the default MLM; ProtBERT is also supported.
- Each model module (`models/esm.py`, `models/esm_validity.py`, `models/prot.py`) exposes a module-level `model_lock`; `score_junctions` and `pseudo_perplexity` hold it around every model call so junction and validity scoring can share the process without racing.
- The repo is installed editable, so imports resolve from the project root.
