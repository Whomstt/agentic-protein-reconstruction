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
      agents/react_agent.py  →  LangGraph ReAct agent (LLM picked by llm_model.profile:
                                 microsoft_foundry or openai_api)
                  │
                  ├── trypsin_filter
                  ├── overlap_graph
                  ├── junction_scorer
                  ├── beam_search
                  └── validity_scorer
```

The important change is that the agent no longer does one pass and stops. It runs for up to `search.max_iterations` rounds, and each round is supposed to try a materially different strategy based on why the previous candidate was weak.

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

3. `junction_scorer(window=None)`

  Scores ordered fragment pairs with a masked language model. The first `W` residues of the successor fragment are masked one at a time and averaged, where `W` defaults to `cfg["mlm_model"]["junction_window"]`.

  The LLM can intentionally vary the window to probe a different local context. The tool can also accept a targeted subset of junction pairs to rescore and merges those results into the existing score matrix in shared state.

4. `beam_search(search_mode="beam", beam_width=None, edge_mode="hard", confirmed_bonus=0.0, window=None)`

  Reconstructs an ordering from the score matrix.

  Supported strategy variations:

  - `search_mode="beam"` or `"greedy"`
  - arbitrary `beam_width`
  - `edge_mode="hard"` or `"soft"`
  - `confirmed_bonus` for soft-rewarding overlap-confirmed edges
  - `window` to deliberately rescore junctions with a different masking window

  Beam search falls back to greedy extension if constraints cut off the search before all fragments are placed.

5. `validity_scorer(reconstruction=None)`

  Computes ESM-2 pseudo-perplexity for a reconstructed sequence. Lower is better.

  The tool scores the explicit reconstruction passed in, or the reconstruction stored in shared state if none is passed.

## Validity Score

The validity score is pseudo-perplexity computed from an ESM-2 masked language model:

- mask one residue at a time,
- get the log probability of the true residue at that position,
- average those log probabilities,
- exponentiate the negative mean.

Lower values mean the reconstruction looks more linguistically plausible to the protein MLM.

Important: this is a plausibility score, not an exact-match oracle. A candidate can have a good validity score and still be wrong in sequence order.

## Configuration

Key config values in [config.yaml](config.yaml):

- `run.method` — `"agentic"` (LLM-driven) or `"sequential"` (deterministic, no LLM); read by `main.py` when `sweep.enabled` is false
- `sweep.enabled` — `false` runs once via `run.method` above; `true` loops the cartesian product of `sweep.grid` (+ `sweep.extra_runs`) instead
- `sweep.grid` — axes to sweep, e.g. `organism`, `replica_count`, `mlm_profile`; each combo gets its own `results/` folder plus one combined cross-combo report
- `data.test_samples` — number of evaluation samples to draw per run (random, see below)
- `data.replica_count` — number of digestion replicas to generate per protein during preprocessing
- `search.max_iterations` — number of iterative agent rounds
- `search.early_stop_patience` — consecutive non-improving iterations before stopping early (set equal to `max_iterations` to disable early stopping, so every sample always runs the full budget)
- `search.beam_width_step` — suggested beam-width adjustment granularity
- `mlm_model.junction_window` — default masking window for junction scoring
- `validity_model.*` — model settings for the validity scorer

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

## Recent Observations

- Replaced the fixed `search.validity_threshold` early-stop gate with a patience-based rule: the run stops once `best_validity_score` fails to improve for `search.early_stop_patience` consecutive iterations, otherwise it runs the full `search.max_iterations`. This removes the need to hand-tune an absolute pseudo-perplexity cutoff per dataset/model.
- The combined sweep report now includes the full per-metric "Iterative Reasoning Gain" table for every combo (previously only exact match/similarity/Kendall tau made it into the cross-combo table; the full baseline→first-pass→best breakdown lived only in each combo's own report).
- The best iteration is often not the first one anymore.
- Exact match is still rare; the validity score is helping with plausibility, not guaranteeing perfect reconstruction.

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
