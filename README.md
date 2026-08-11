# Agentic De Novo Protein Reconstruction

Reconstructing protein sequences from unordered trypsin-digestion fragments. An LLM
agent iterates over reconstruction strategies, scores each candidate with an
ESM-2 based validity signal, and keeps the best one. Unlike most protein design
work, which builds novel sequences with intended properties, this reassembles a
sequence that already exists from its fragments.

Each protein is run for several rounds. Every round tries a materially different
strategy, chosen from five fixed levers, based on why the previous candidate
scored poorly. Built on LangChain / LangGraph with a protein language model
(ESM-2 by default, ProtBERT optional) as the scoring engine.

## Setup

Dependencies are managed with [uv](https://docs.astral.sh/uv/) (see
`pyproject.toml` / `uv.lock`; Python 3.13, CUDA 12.2 torch build):

```bash
uv sync
```

Create a `.env` in the project root with credentials for your LLM provider:

```bash
# Azure Foundry (llm_model.profile: microsoft_foundry)
AZURE_ENDPOINT=...
AZURE_API_KEY=...          # omit to use DefaultAzureCredential (Azure AD) instead

# OpenAI API (llm_model.profile: openai_api)
OPENAI_API_KEY=...
```

Only the `agentic` path needs an API key; `sequential` runs without one. Protein
language model weights download from HuggingFace on first use.

## Data

Fragmented datasets are generated from the reviewed (Swiss-Prot) UniProt FASTA
(https://www.uniprot.org/help/downloads) placed at `data/raw/`. `python main.py`
regenerates the active organism's dataset automatically whenever
`data.organism`, `data.replica_count` or `data.missed_cleavage_ratio` has changed
since it was last written; run `python -m preprocessing.preprocessing` only to
force a rebuild.

## Usage

`python main.py` is the single entry point; everything is controlled by
[config.yaml](config.yaml):

- `run.method: "agentic"` or `"sequential"` — reconstruction approach, used when
  `sweep.enabled` is `false`.
- `sweep.enabled: true` — loop every combination in `sweep.grid` (e.g. organism ×
  replica_count × PLM profile), each as its own subprocess, then write one
  combined cross-combo report on top of each combo's own report.

Run one evaluation mode directly, bypassing `run.method`:

```bash
python -m evaluation.agentic          # agentic evaluation only
python -m evaluation.sequential       # deterministic baseline only
python -m evaluation.junction_ranking # search-independent junction-ranking diagnostic
```

## Results

Every run writes a timestamped folder under `results/` with `report.md`,
`metric_comparison.svg`, `summary.json` and `samples.jsonl` (full per-sample,
per-iteration detail), plus a statistical layer: `analysis_report.md`,
`results.csv`, `summary.csv`, `tables/*.tex` and `figures/*.pdf`.

An agentic `report.md` leads with a paired benchmark of **Shuffled Baseline →
Deterministic → Agentic Best**. With `run.control_baseline.enabled` it adds a
matched-budget non-LLM **Control** arm and a `Δ Agentic − Control` column, the
fair test of whether the LLM's decision-making helps: the iteration budget, tool
pipeline and selection rule are held fixed so only the lever source differs. With
`run.report_oracle` it adds an **Oracle** ceiling (the best candidate the agent
actually generated). It also reports **Validity Signal Concordance** and **Cost,
Efficiency & Completion**.

The reporting layer runs offline, with no GPU, model or network, so it can be
regenerated from stored data at any time:

```bash
python -m evaluation.rebuild --all                  # reports, CSVs, tables, figures
python -m evaluation.thesis_tables --run <folder>   # the paper's tables -> report/tables/
```

`results/` is gitignored, so the six reported runs are published under
[final_results/](final_results/) instead — two organisms × three replica counts,
600 samples, ~8 MB. Each folder keeps its full raw per-sample record
(`samples.jsonl.gz`) alongside the reports, CSVs, tables and figures, so every
reported number can be recomputed with `--results-root final_results`. See
[final_results/README.md](final_results/README.md).

Metrics account for the fact that a reconstruction is a permutation of a fixed
fragment set: **Exact Match**, **Sequence Similarity**, **Adjacent Pair
Accuracy**, **Longest Correct Run** and **Kendall Tau**. See
[CLAUDE.md](CLAUDE.md#metrics) for definitions.

## Agent runtime

- Switch LLMs via `llm_model.profile` (`openai_api` or `microsoft_foundry`).
- Switch PLMs via `mlm_model.profile` (`esm_small`, `esm_medium`, `protbert`);
  matching junction and validity settings are derived automatically.
- The agent's controllable levers are fixed to five: junction masking window,
  search mode, beam width, edge mode and confirmed-edge bonus.

See [CLAUDE.md](CLAUDE.md) for the architecture, evaluation arms and
research-validity notes.

## Project structure

```
main.py            Single entry point; dispatches on config.yaml
config.py          Loads config.yaml, resolves model/dataset profiles, seeds RNG
config.yaml        All runtime configuration

agents/            Iterative loop, single-call/react drivers, control-arm policy
algorithms/        Pure computation (scoring, ordering, overlap graph)
tools/             LangChain @tool wrappers + shared run-time state
models/            HuggingFace loaders for ESM-2 / ProtBERT and memory helpers
evaluation/        Sample loop, metrics, statistics, reporting, sweep orchestration
preprocessing/     FASTA filtering and trypsin digestion into fragmented datasets
tests/             stdlib unittest suites (no pytest dependency)

experiments/       Exploratory notebooks (not part of the pipeline)
final_results/     The six reported runs, committed (results/ itself is gitignored)
report/            LaTeX write-up, generated tables and figures
```
