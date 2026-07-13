# agentic-protein-reconstruction
## Agentic De Novo Protein Reconstruction from Fragmented Sequence Mixtures

### Overview
This research is tackling the challenge of de novo protein reconstruction from fragmented sequence mixtures found during real-world experimentation. Using LangChain and LangGraph to architect an agentic AI framework, and making use of Protein Language Models as the biological reasoning engine, the system autonomously navigates the reconstruction process guided by learned biochemical constraints from curated protein databases. Most research in the field currently involves constructing proteins from scratch with intended properties. We take it a step back to reconstruct that which already exists, just waiting to be pieced back together.

The agent runs for multiple rounds per sample, trying a materially different reconstruction strategy each time based on why the previous candidate scored poorly, and keeps whichever candidate scores best on an ESM-2 plausibility (validity) model.

### Dataset
- Reviewed (Swiss-Prot) FASTA: https://www.uniprot.org/help/downloads
- `python main.py` regenerates the fragmented dataset for you automatically whenever `data.organism`, `data.replica_count`, or `data.missed_cleavage_ratio` has changed since it was last written — no manual step needed. Run `python -m preprocessing.preprocessing` directly only if you want to force a rebuild.

### Run it
`python main.py` is the single entry point; everything is controlled by [config.yaml](config.yaml):

- `run.method: "agentic"` or `"sequential"` — which reconstruction approach to run (used when `sweep.enabled` is `false`)
- `sweep.enabled: true` — instead of one run, loop every combination in `sweep.grid` (e.g. organism × replica_count × PLM profile), each as its own subprocess, then write one combined report across all combos on top of each combo's own report

Every run writes a timestamped folder under `results/` with `report.md`, `metric_comparison.svg`, `summary.json`, and `samples.jsonl` (full per-sample, per-iteration detail). An agentic `report.md` leads with a paired benchmark of **Shuffled Baseline → Deterministic → Agentic Best**. With `run.control_baseline.enabled` it also carries a matched-budget non-LLM **Control** arm and a `Δ Agentic − Control` column — the fair test of whether the LLM's decision-making helps, holding the iteration budget, tool pipeline, and selection rule fixed so only the lever source differs. With `run.report_oracle` it adds an **Oracle** ceiling (the best candidate the agent actually generated). It also reports **Validity Signal Concordance** (whether the selection signal tracks true quality) and **Cost, Efficiency & Completion** (LLM calls/tokens/wall-clock, completion and failure rates) — the axes that quality alone can't justify the agent on.

Metrics are tuned for the fact that a reconstruction is a permutation of a fixed fragment set: **Exact Match**, **Sequence Similarity** (the one soft string metric), **Adjacent Pair Accuracy** and **Longest Correct Run** (ordering / partial-assembly), and **Kendall Tau** (global ordering). See [CLAUDE.md](CLAUDE.md#metrics) for the definitions.

To run one evaluation mode directly, bypassing `run.method`:
```bash
python -m evaluation.agentic
python -m evaluation.sequential
python -m evaluation.junction_ranking   # search-independent check that the pLM ranks true successor fragments well
```

### Agent runtime
- Switch LLMs by changing `llm_model.profile` in [config.yaml](config.yaml). Use `openai_api` for the OpenAI API and `microsoft_foundry` for Azure Foundry.
- Switch PLMs by changing `mlm_model.profile` in [config.yaml](config.yaml); the matching junction and validity settings are derived automatically.
- The Microsoft Foundry profile reads `AZURE_ENDPOINT` and prefers `AZURE_API_KEY` when present; otherwise it falls back to Azure AD token auth via `DefaultAzureCredential`.
- The agent's controllable strategy levers are fixed to five: junction masking window, search mode, beam width, edge mode, and confirmed-edge bonus. See [CLAUDE.md](CLAUDE.md) for the full architecture.
