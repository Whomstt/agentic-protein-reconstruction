import json
from agents.react_agent import build_agent
from evaluation.metrics import METRIC_NAMES, compute_all, print_metrics, print_comparison
from config import cfg


def extract_reconstruction(result):
    """Extract reconstruction and order from the agent's final tool call."""
    for msg in reversed(result["messages"]):
        if hasattr(msg, "name") and msg.name == "beam_search":
            content = (
                json.loads(msg.content) if isinstance(msg.content, str) else msg.content
            )
            return content["reconstruction"], content["order"]
    return result["messages"][-1].content, []


test_path = cfg["data"]["ecoli_test_split"]
sample_count = 10  # set to None to evaluate all samples

with open(test_path) as f:
    samples = [json.loads(l) for l in f if l.strip()][:sample_count]

agent = build_agent()

baseline_summary = {k: [] for k in METRIC_NAMES}
recon_summary = {k: [] for k in METRIC_NAMES}

print(f"Agentic Evaluation ({len(samples)} Samples)")
print("-" * 60)

for i, sample in enumerate(samples, 1):
    target = sample["ecoli_original"]
    fragments = sample["fragments"]

    # Baseline: fragments in their (shuffled) input order
    baseline_order = list(range(len(fragments)))
    baseline_recon = "".join(fragments)
    baseline_metrics = compute_all(target, baseline_recon, fragments, baseline_order)

    # Reconstructed: agent output
    result = agent.invoke(
        {
            "messages": [
                ("user", f"Reconstruct the protein from these fragments: {fragments}")
            ]
        }
    )
    reconstruction, order = extract_reconstruction(result)
    recon_metrics = compute_all(target, reconstruction, fragments, order)

    for k in METRIC_NAMES:
        baseline_summary[k].append(baseline_metrics[k])
        recon_summary[k].append(recon_metrics[k])

    print(f"Sample {i}")
    print(f"  Target:         {target}")
    print(f"  Reconstruction: {reconstruction}")
    print_metrics(recon_metrics)

if samples:
    n = len(samples)
    print(f"\nAverage Results ({n} Samples) — Shuffled vs Reconstructed")
    print("-" * 60)
    print(f"  LLM Agent: {cfg['llm_model']['name']}")
    print(f"  PLM Model: {cfg['mlm_model']['name']}")
    print(f"  Beam Size: {cfg['mlm_model'].get('beam_size', 'N/A')}")
    print(f"  Missed Cleavage Ratio: {cfg['data'].get('missed_cleavage_ratio', 'N/A')}")
    print(f"  Minimum Peptide Length: {cfg['data'].get('min_length', 'N/A')}")
    print()
    print_comparison(baseline_summary, recon_summary, n)
