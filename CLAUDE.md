# Agentic Protein Reconstruction

Reconstruct protein sequences from unordered digestion fragments using an LLM agent that iterates over multiple reconstruction hypotheses, scores each one with an ESM-2 validity model, and keeps the best candidate.

## Current Architecture

```
main.py / evaluation/agentic.py
      │
      ▼
agents/iterative_runner.py
      │
      ▼
agents/react_agent.py  →  LangGraph ReAct agent (GPT-5.4-mini)
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

All other experiment controls are fixed and off-limits to the agent, including `search.validity_threshold`, `search.max_iterations`, PLM/model choice, `max_length`, `batch_size`, and replica selection.

## System Layers

- `tools/` — LangChain `@tool` wrappers. They are thin adapters around the algorithms and store run-time artifacts in `tools/state.py`.
- `algorithms/` — Pure computation. No LangChain dependencies.
- `models/` — HuggingFace model loaders for ESM-2 or ProtBERT. Junction scoring and validity scoring use shared model instances and reset the ESM rotary cache between calls.
- `agents/iterative_runner.py` — Orchestrates the multi-iteration loop, prompts the LLM with the previous validity score and strategy, and stores per-iteration history.

## Iterative Agent Loop

Each iteration builds a prompt that explicitly tells the LLM to:

- explain why the previous attempt likely failed,
- choose a different tactic using only the five levers above,
- use targeted junction rescoring when only a few pairs need fresh scores,
- run the needed tools for a fresh candidate,
- call `validity_scorer` on the candidate, and
- stop early only if the validity score is at or below `search.validity_threshold`.

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

- `data.test_samples` — number of evaluation samples to run in sequential/agentic evaluation
- `data.replica_count` — number of digestion replicas to generate per protein during preprocessing
- `search.max_iterations` — number of iterative agent rounds
- `search.validity_threshold` — early-stop gate for pseudo-perplexity
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

The preprocessing pipeline keeps all surviving fragments and now generates `replica_count` digestion replicas per protein.

`target_reconstruction` is the ground truth target for both evaluation modes.

## Evaluation

- `evaluation/sequential.py` — deterministic baseline using the algorithm pipeline directly.
- `evaluation/agentic.py` — runs the iterative agent on each sample, then scores the best reconstruction it found.
- `evaluation/metrics.py` — shared metrics: `exact_match`, `similarity`, `fragment_acc`, `norm_edit_distance`, `lcs_ratio`, `adjacent_pair_acc`, `kendall_tau`.
- `evaluation/reporting.py` — prints config snapshots, per-sample results, and summary tables.

The agentic evaluation stores the full iteration history per sample in the results payload.

## Recent Observations

- Lowering `search.validity_threshold` from 25.0 to 12.0 made later iterations actually run on some samples.
- The best iteration is often not the first one anymore.
- Exact match is still rare; the validity score is helping with plausibility, not guaranteeing perfect reconstruction.
- The current 5-sample agentic run improved over shuffled baseline on all reported metrics, but it did not make exact reconstruction easy.

## Commands

```bash
python main.py
python -m preprocessing.preprocessing
python -m evaluation.sequential
python -m evaluation.agentic
```

## Conventions

- No `__init__.py` files; imports are flat.
- All config access goes through `from config import cfg`.
- Algorithms stay pure; tools manage state.
- The project uses the ESM-2 MLM by default, but ProtBERT remains supported.
- The repo is installed editable, so imports resolve from the project root.
