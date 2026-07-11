# Agentic Protein Reconstruction

Reconstruct protein sequences from unordered digestion fragments using an LLM agent that iterates over multiple reconstruction hypotheses, scores each one with an ESM-2 validity model, and keeps the best candidate.

## Current Architecture

```
main.py
      │
      ├── sweep.enabled: true  → evaluation/sweep.py loops config.yaml's sweep.grid,
      │                          each combo re-running main.py as a subprocess
      │
      └── sweep.enabled: false → evaluation/runner.py::run_agentic() or run_sequential()
                  │             (picked by run.method)
                  ▼
      agents/iterative_runner.py
                  │
                  ▼
      agents/react_agent.py  →  build_agent() (LLM picked by llm_model.profile:
                                 microsoft_foundry or openai_api)
                  │
                  ├── run.calling_mode: "react" (default)
                  │     LangGraph ReAct agent drives every tool call itself
                  │     (~4-6 LLM calls/iteration: one per tool-call decision)
                  │
                  └── run.calling_mode: "single_call"
                        agents/deterministic_agent.py — 1 LLM call/iteration
                        picks only the 5 lever values (structured output);
                        the harness runs the fixed tool pipeline in Python
                  │
                  ├── trypsin_filter
                  ├── overlap_graph
                  ├── junction_scorer
                  ├── beam_search
                  └── validity_scorer
```

The important change is that the agent no longer does one pass and stops. It runs for up to `search.max_iterations` rounds, and each round is supposed to try a materially different strategy based on why the previous candidate was weak.

`run.calling_mode` controls how much the LLM is in the driver's seat. `"react"` lets the LLM decide every individual tool call (`trypsin_filter`, `overlap_graph`, `junction_scorer`, `beam_search`, `validity_scorer` are each separate LLM turns via LangGraph's `create_react_agent`), which costs several LLM calls per iteration even though most of that sequence is mechanically fixed by the system prompt. `"single_call"` keeps the same five levers and iteration-history prompting but calls the LLM at most once per iteration — via `agents/deterministic_agent.py`'s `LeverChoice` structured output — for `junction_window`, `search_mode`, `beam_width`, `edge_mode`, `confirmed_bonus`; the harness (`agents/iterative_runner.py::run_iterative_reconstruction`, dispatching on `agent.mode`) then runs the tool pipeline directly in Python with those values. With `run.iteration1_deterministic=true` (default), iteration 1 uses the fixed config defaults (`_default_lever_values()`) with **no LLM call at all**, since there's no prior attempt yet to react to — a 3-iteration run then costs 2 LLM calls/sample in `single_call` mode, not 3. Set `run.iteration1_deterministic=false` to make iteration 1 a genuine LLM lever choice too (3 LLM calls/sample) — see the Research Validity Notes section for why this matters for reporting. `trypsin_filter`/`overlap_graph` run once on iteration 1 with no LLM involvement in either mode. `single_call` mode also skips the explicit `junction_scorer` rescore on iterations 2+ when the LLM keeps `junction_window` unchanged from the previous iteration — it reuses `state["scores"]` and lets `beam_search`'s own lazy-rescore check confirm no recompute is needed, avoiding a redundant full MLM pass over every unscored junction pair when only `search_mode`/`beam_width`/`edge_mode`/`confirmed_bonus` changed. `agents/react_agent.py::build_agent()` returns a mode-tagged `Agent` (`.graph` for react, `.llm` for single_call) so `evaluation/runner.py` doesn't need to know which mode is active.

The finalized agent control surface is limited to five levers only:

- junction masking window via `junction_scorer(window=...)` or `beam_search(window=...)`
- search mode via `beam_search(search_mode="beam"|"greedy")`
- beam width via `beam_search(beam_width=...)`
- edge mode via `beam_search(edge_mode="hard"|"soft")`
- confirmed-edge bonus via `beam_search(confirmed_bonus=...)`

All other experiment controls are fixed and off-limits to the agent, including `search.early_stop_patience`, `search.max_iterations`, PLM/model choice, `max_length`, `batch_size`, and replica selection.

## System Layers

- `tools/` — LangChain `@tool` wrappers. They are thin adapters around the algorithms and store run-time artifacts in `tools/state.py`.
- `algorithms/` — Pure computation. No LangChain dependencies.
- `models/` — HuggingFace model loaders for ESM-2 or ProtBERT. Junction scoring and validity scoring use shared model instances and reset the ESM rotary cache between calls.
- `agents/iterative_runner.py` — Orchestrates the multi-iteration loop, prompts the LLM with the previous validity score and strategy, and stores per-iteration history.
- `evaluation/runner.py` — Single source of truth for running an evaluation over the active test split (both `run_sequential` and `run_agentic`); `main.py`, `evaluation/sequential.py`, and `evaluation/agentic.py` all call into it instead of duplicating the sample loop.
- `evaluation/sweep.py` — Loops every combination in `sweep.grid` (+ `sweep.extra_runs`), overriding `data.organism`/`data.replica_count`/`mlm_model.profile` per combo and running `python -m main` as its own subprocess against a generated override config (`AGENTIC_CONFIG_PATH`). The checked-in `config.yaml` is never modified.
- `evaluation/sweep_report.py` / `evaluation/sweep_pdf.py` — After a sweep finishes, build one combined `report.md`/`report.pdf` across all combos: a quick-glance comparison table plus a full per-metric "Iterative Reasoning Gain" table (baseline → first-pass → best) for every combo, pulled from each combo's own `summary.json`.

## Iterative Agent Loop

Each iteration builds a prompt that explicitly tells the LLM to:

- explain why the previous attempt likely failed,
- choose a different tactic using only the five levers above,
- use targeted junction rescoring when only a few pairs need fresh scores,
- run the needed tools for a fresh candidate,
- call `validity_scorer` on the candidate, and
- stop early once the best validity score hasn't improved for `search.early_stop_patience` consecutive iterations, otherwise run until `search.max_iterations`.

Per-iteration results now record `lever_values` and `changed_levers` so runs remain auditable after the fact.

**Best-candidate selection is hysteretic, not raw argmin.** Iteration 1's candidate (canonical strategy) is the incumbent; a later iteration only replaces it when its validity score is lower by more than `search.improvement_margin` (relative). The validity signal is the junction+overlap blend above (`algorithms/score_validity.blended_validity`), a much more reliable ordering signal than the old whole-sequence pseudo-perplexity but still imperfect (~57–61% concordant with true quality), so raw argmin over a few deliberately-diverse candidates is still a winner's curse that can swap to a noise-lower candidate. The margin keeps iteration 1 unless a later round clearly beats it, so the iterated result is `>=` the first pass on average.

The shared state records:

- `fragment_samples`
- `fragments`
- `iteration_history`
- `best_iteration`
- `best_reconstruction`
- `best_validity_score`
- `best_order`
- `search_strategy`
- `reconstruction`
- `order`
- `validity_score`

## Tools

1. `trypsin_filter(fragments)`

  Initializes trypsin-derived constraints and beam hints.

  Outputs and state keys include:

  - `impossible_junctions`
  - `missed_cleavage_fragments`
  - `start_candidates`

2. `overlap_graph(fragment_samples)`

  Builds hard adjacency edges from multi-sample digestions and stores:

  - `confirmed_junctions`
  - `confirmed_adjacencies`
  - `confirmed_successors`
  - `unscored_junctions`

3. `junction_scorer(window=None, junction_pairs=None)`

  Scores ordered fragment pairs with a masked language model. The first `W` residues of the successor fragment are masked one at a time and averaged, where `W` defaults to `search.default_levers.junction_window`.

  The LLM can intentionally vary the window to probe a different local context. The tool can also accept a targeted subset of junction pairs to rescore and merges those results into the existing score matrix in shared state.

  **Feedback signal:** returns `mean_score`/`min_score`/`max_score`/`num_junctions_scored` over the pairs just scored. A narrow spread near the mean means the current window isn't giving the PLM enough signal to discriminate real junctions from wrong ones — the next iteration should try a different window rather than proceeding with weak scores.

4. `beam_search(search_mode="beam", beam_width=None, edge_mode="hard", confirmed_bonus=0.0, window=None)`

  Reconstructs an ordering from the score matrix.

  Supported strategy variations:

  - `search_mode="beam"` or `"greedy"`
  - arbitrary `beam_width`
  - `edge_mode="hard"` or `"soft"`
  - `confirmed_bonus` for soft-rewarding overlap-confirmed edges
  - `window` to deliberately rescore junctions with a different masking window

  Beam search falls back to greedy extension if constraints cut off the search before all fragments are placed; greedy search itself falls back to the least-bad remaining candidate if every option at a step is trypsin-impossible (previously this could silently duplicate a fragment in the order — fixed alongside the diagnostic below).

  **Feedback signal:** returns `fell_back` (beam mode: true means `beam_width`/`edge_mode` cut the search off short — widen the beam or try `edge_mode="soft"`), `forced_impossible_count` (greedy mode: non-zero means trypsin constraints fought the ordering — try `search_mode="beam"`), `num_confirmed_edges_realized`/`num_confirmed_edges_total` (how many overlap-confirmed adjacencies made it into the final order), and `mean_junction_score` (a cheap preview of ordering quality before paying for `validity_scorer`).

5. `validity_scorer(reconstruction=None)`

  Scores the ordering currently in shared state as the basis for best-candidate selection. It combines two junction-focused signals (see `algorithms/score_validity.blended_validity_detailed`) rather than whole-sequence pseudo-perplexity:

  - **junction-local pseudo-perplexity** — masked-LM plausibility of only the fragment junctions the ordering uses that the overlap graph did *not* already confirm (reuses the junction-scoring model on the order's non-confirmed consecutive pairs, fixed `search.validity_junction_window`); confirmed junctions are skipped here since they're already known-valid from real multi-replica overlap evidence rather than a model guess, and
  - **confirmed-adjacency agreement** — how well the ordering respects the overlap graph's confirmed adjacencies (`state["confirmed_junctions"]`), applied as a multiplicative penalty weighted by `search.validity_confirmed_penalty`.

  If an explicit reconstruction different from state's is passed (its ordering is unknown), the tool falls back to whole-sequence pseudo-perplexity on that string.

  **Feedback signal:** returns a dict, not a bare float — `validity_score` (lower is better; this is what decides which iteration's candidate wins) plus its two components `junction_local_ppl` and `confirmed_adjacency_agreement`, and `confirmed_penalty_applied`. Exposing the components (not just the blended score) lets the next iteration's lever choice target the actual weak point: high `junction_local_ppl` → change `junction_window`; low `confirmed_adjacency_agreement` → switch `edge_mode` to `"hard"` or raise `confirmed_bonus` in `"soft"` mode.

  **Propagation across modes:** in `react` mode the LLM sees these dicts directly in tool messages as it reasons live. In `single_call` mode (see `run.calling_mode`), the harness never lets the LLM see raw tool output, so `agents/deterministic_agent.py::_lever_prompt` explicitly threads the previous iteration's `validity_breakdown`, `junction_stats`, and `beam_diagnostics` into the next iteration's prompt text — without this, `single_call` mode would only know the previous blended score and lever values, not *why* it scored that way.

## Validity Score (best-candidate selection signal)

Candidate orderings reuse the same fragments and differ only at the junctions, so the selection signal scores only what varies. It is:

```
junction_local_ppl * (1 + confirmed_penalty * (1 - confirmed_adjacency_agreement))
```

- **junction_local_ppl** = `exp(-mean_junction_logprob)` over the ordering's non-confirmed boundaries only (see below). Whole-sequence pseudo-perplexity is ~95% invariant across orderings (the within-fragment residues are identical), which drowns the ordering signal. Scoring only the junctions keeps the residues whose likelihood actually depends on the order.
- **confirmed_adjacency_agreement** = fraction of the overlap graph's confirmed directed edges realized as consecutive in the ordering. A near-ground-truth structural signal (real multi-replica overlaps) that strengthens with replica count.

The two hard-constraint tools feed this signal in different ways rather than being re-scored themselves:
- `trypsin_filter`'s `impossible_junctions` (K/R→P and non-K/R-terminal violations) are excluded from every candidate ordering by construction in `beam_order`/`greedy_order` regardless of lever settings, so they can never appear in an ordering and need no separate validity check.
- `overlap_graph`'s `confirmed_junctions` are excluded from the PLM-scored junction set in `junction_local_ppl` (they're already known-valid from real multi-replica overlap, not a model guess) and instead scored via `confirmed_adjacency_agreement`.

Offline validation (junction window 5, confirmed penalty 0.75): concordance with true reconstruction quality is ~57% on yeast and ~61% on E. coli, and it flips the iterated-vs-first-pass result on yeast from worse (−0.026) to better (+0.006).

Important: this is still a plausibility/consistency score, not an exact-match oracle. A candidate can score well and still be wrong in sequence order.

## Configuration

Key config values in [config.yaml](config.yaml):

- `run.method` — `"agentic"` (LLM-driven) or `"sequential"` (deterministic, no LLM); read by `main.py` when `sweep.enabled` is false
- `sweep.enabled` — `false` runs once via `run.method` above; `true` loops the cartesian product of `sweep.grid` (+ `sweep.extra_runs`) instead
- `sweep.grid` — axes to sweep, e.g. `organism`, `replica_count`, `mlm_profile`; each combo gets its own `results/` folder plus one combined cross-combo report
- `data.test_samples` — number of evaluation samples to draw per run (random, see below)
- `data.replica_count` — number of digestion replicas to generate per protein during preprocessing
- `search.max_iterations` — number of iterative agent rounds
- `run.iteration1_deterministic` — **research-validity-relevant, see below.** `true` (default): iteration 1 uses `search.default_levers` directly with no LLM call, saving one LLM call/sample; `false`: iteration 1 is also a genuine `LeverChoice` LLM decision, like every other round. Only affects `single_call` mode (`react` mode always drives every tool call, including iteration 1's, through the LLM). Reports/labels adjust automatically: `evaluation/reporting.py::first_pass_label()` renders "1st Iteration (Config Defaults)" when `true` or "1st Iteration (LLM)" when `false`, so the config/gain tables never silently imply agentic behavior that didn't happen.
- `search.default_levers` — the single source of truth for all five agent-controllable levers (`junction_window`, `search_mode`, `beam_width`, `edge_mode`, `confirmed_bonus`) across the whole pipeline, not just the agent. When `run.iteration1_deterministic` is `true`, iteration 1 uses these directly with no LLM call; in `react` mode, they're the fallback whenever the LLM's tool call omits a lever argument (`agents/deterministic_agent.py::_default_lever_values()` / `agents/iterative_runner.py::DEFAULT_LEVER_VALUES`). `junction_window`/`beam_width` are also the fallback read directly by `algorithms/score_junctions.py`, `algorithms/beam_order.py`, `tools/junction_scorer.py`, `tools/beam_search.py` whenever no explicit value is passed — including `run_sequential`, which has no agent at all. `mlm_model.profile` no longer carries its own `junction_window`/`beam_width` (removed to avoid two config values silently drifting out of sync); only `search.default_levers` controls them now.
- `search.early_stop_patience` — consecutive non-improving iterations before stopping early (set equal to `max_iterations` to disable early stopping, so every sample always runs the full budget)
- `search.beam_width_step` — a numeric hint injected into the lever-choice prompt ("adjust beam_width by roughly N at a time"), not an enforced constraint — the LLM's returned `beam_width` is passed through unmodified regardless of this value (no code clamps or rounds to it). It exists to bias the LLM away from erratic large jumps, at the cost of nudging its exploration granularity rather than letting it infer a step size purely from diagnostics. Disclose this as a prompt-engineering choice if reporting on lever diversity/exploration behavior.
- `search.improvement_margin` — minimum *relative* validity drop a later iteration must clear to replace the incumbent best (default `0.030`, tuned on a small offline set to keep the iterated result `>=` first-pass on **both** ecoli and yeast; lower values ~0.010–0.015 give a bigger yeast gain but let ecoli regress). Guards against winner's-curse swaps on validity noise (see Iterative Agent Loop); `0.0` restores raw-argmin selection. **Leakage risk, disclosed rather than fixed:** there is no train/test split anywhere in this project (see Data section) — the 15 ecoli / 20 yeast offline tuning set this value was chosen on was drawn from the same undivided per-organism pool that `data.test_samples`/`sweep.test_samples` also draw evaluation samples from, so this constant was not tuned on data disjoint from what gets reported. Treat `0.030` as a documented sensitivity choice, not a validated hyperparameter, in any paper/writeup; a properly disjoint tune/eval split would need a held-out partition of the fragmented pool (e.g. by protein) before re-tuning could be trusted.
- `search.validity_junction_window` — residues of each junction scored for the validity/selection signal (default `5`)
- `search.validity_confirmed_penalty` — weight on `(1 - confirmed_adjacency_agreement)` in the validity signal; inflates junction PPL when the ordering violates overlap-confirmed edges (default `0.75`)
- `validity_model.*` — model settings used only for the whole-sequence pseudo-perplexity fallback (when an explicit reconstruction with no known ordering is scored)

## Data

Data is JSONL with fields like:

- `fragments`
- organism-specific originals such as `ecoli_original` and `yeast_original`
- `target_reconstruction`
- `num_fragments`
- `replica_count`
- `missed_cleavage_ratio`

`preprocessing/preprocessing.py` filters the raw UniProt fasta to the active organism, deduplicates by gene (`GN=` tag) so the same protein isn't repeated across near-identical bacterial strains, then generates `replica_count` digestion replicas per protein. There is no train/test split — no training happens in this project, so preprocessing writes one deduped fragmented file per organism (`data.fragmented_ecoli` etc.) and evaluation draws directly from it.

Each fragmented output has a sidecar `.meta.json` recording the `organism`/`replica_count`/`missed_cleavage_ratio` it was generated with. `preprocessing.preprocessing.ensure_fresh_dataset()` compares that against the active config and regenerates the dataset if any of the three changed; `main.py` calls this automatically before every non-sweep run, so editing `data.replica_count` (or organism, or missed-cleavage ratio) and just running `python main.py` picks it up without a manual `python -m preprocessing.preprocessing` step. Each sweep combo re-enters `main.py` as its own subprocess against its per-combo config, so this check naturally re-runs preprocessing once per distinct combination in `sweep.grid` (e.g. once per `replica_count` value per organism), not on every combo.

`evaluation/runner.py::_load_test_samples` shuffles that pool (using the global `misc.seed`, set once in `config.py` for `random`/`numpy`/`torch`) and takes the first `data.test_samples` records. If `test_samples` is unset, the whole pool is used.

`target_reconstruction` is the ground truth target for both evaluation modes.

## Evaluation

- `evaluation/runner.py` — shared sample loop for both `run_sequential` (deterministic baseline) and `run_agentic` (iterative agent, then scores the best reconstruction found, plus a first-pass/iteration-1 comparison to isolate what refinement added).
- `evaluation/sequential.py` / `evaluation/agentic.py` — thin CLI wrappers around `run_sequential`/`run_agentic`, independent of `run.method`.
- `evaluation/metrics.py` — shared metrics: `exact_match`, `similarity`, `fragment_acc`, `norm_edit_distance`, `lcs_ratio`, `adjacent_pair_acc`, `kendall_tau`.
- `evaluation/reporting.py` — per-run config snapshots, per-sample results, distribution stats, and the SVG charts/markdown/PDF report for a single run.
- `evaluation/sweep.py` + `evaluation/sweep_report.py` + `evaluation/sweep_pdf.py` — grid sweep orchestration and the combined cross-combo report.

Each run's `results/<timestamp>_<name>/` folder contains `summary.json`, `samples.jsonl` (full per-sample + per-iteration history for auditability), `report.md`, `report.pdf`, and chart SVGs. The agentic evaluation stores the full iteration history (including `lever_values`/`changed_levers`) per sample in the results payload.

## Research Validity Notes

Audited 2026-07-11 for whether the LLM's lever choices are genuine evidence-based decisions rather than scripted behavior, ahead of any research/paper claims. Findings:

- **No ground-truth leakage (clean).** `target_reconstruction` is only read in `evaluation/runner.py::run_agentic()` after `run_iterative_reconstruction()` returns, purely for scoring — it is never passed into `agents/iterative_runner.py`, `agents/deterministic_agent.py`, `agents/react_agent.py`, or any tool. The LLM never sees the true sequence.
- **No hidden clamping or best-of-N.** Whatever the LLM (or, in `single_call` mode, the `LeverChoice` structured output) returns for the five levers is passed straight through to the tools with no post-hoc override, and there is no silent retry/resample that keeps only a favorable LLM call.
- **`run.iteration1_deterministic`** and **`search.improvement_margin`** are the two settings requiring explicit disclosure in any writeup — see their entries in Configuration above for what to state.
- **`search.beam_width_step`** is prompt-only guidance, never enforced — also worth a one-line disclosure if characterizing the agent's exploration behavior.

## Recent Observations

- Replaced the fixed `search.validity_threshold` early-stop gate with a patience-based rule: the run stops once `best_validity_score` fails to improve for `search.early_stop_patience` consecutive iterations, otherwise it runs the full `search.max_iterations`. This removes the need to hand-tune an absolute pseudo-perplexity cutoff per dataset/model.
- The combined sweep report now includes the full per-metric "Iterative Reasoning Gain" table for every combo (previously only exact match/similarity/Kendall tau made it into the cross-combo table; the full baseline→first-pass→best breakdown lived only in each combo's own report).
- The best iteration is often not the first one anymore.
- Exact match is still rare; the validity score is helping with plausibility, not guaranteeing perfect reconstruction.
- Raw-argmin selection over iterations was a winner's curse: because whole-sequence pseudo-perplexity barely varies with fragment order (only the ~few junction positions differ), picking the lowest score across diverse candidates sometimes chose a noise-lower-but-worse ordering, so the iterated result underperformed the first pass on some combos. Added `search.improvement_margin` (hysteresis): a later iteration must beat the incumbent by a relative margin to be adopted. On the r5/r20/r100 ecoli sweep this moved the iterated result from mixed (worse than the 1st iteration on the r20 combo) to `>=` the first pass on ~6/7 metrics in aggregate.

## Commands

```bash
python main.py                       # single entry point; behavior fully controlled by config.yaml
python -m preprocessing.preprocessing # regenerate the fragmented dataset for the active organism
python -m evaluation.sequential       # deterministic baseline only, bypassing run.method
python -m evaluation.agentic          # agentic evaluation only, bypassing run.method
```

## Conventions

- No `__init__.py` files; imports are flat.
- All config access goes through `from config import cfg`.
- Algorithms stay pure; tools manage state.
- The project uses the ESM-2 MLM by default, but ProtBERT remains supported.
- The repo is installed editable, so imports resolve from the project root.
