import json
from difflib import SequenceMatcher
from tools.reconstruction_tool import reconstruct_tool
from config import cfg


test_path = cfg["data"]["ecoli_test_split"]
sample_count = 100  # set to None to evaluate all samples

with open(test_path) as f:
    samples = [json.loads(l) for l in f if l.strip()][:sample_count]


def count_correctly_placed_fragments(original, fragments, order):
    cursor = 0
    correct = 0
    for idx in order:
        frag = fragments[idx]
        if original[cursor : cursor + len(frag)] == frag:
            correct += 1
        cursor += len(frag)
    return correct


summary = {
    "similarity": [],
    "residue_acc": [],
    "fragment_acc": [],
}

print(f"Reconstruction Evaluation ({len(samples)} Samples)")
print("-" * 60)

for i, sample in enumerate(samples, 1):
    original_key = next(k for k in sample if k.endswith("_original"))
    original = sample[original_key]
    fragments = sample["fragments"]

    result = reconstruct_tool.invoke({"fragments": fragments})
    reconstruction, order = result["reconstruction"], result["order"]

    sim = SequenceMatcher(None, original, reconstruction).ratio()

    max_len = max(len(original), len(reconstruction))
    correct_residues = sum(
        a == b for a, b in zip(original.ljust(max_len), reconstruction.ljust(max_len))
    )
    residue_acc = correct_residues / max_len if max_len else 0.0

    correctly_placed = count_correctly_placed_fragments(original, fragments, order)
    fragment_acc = correctly_placed / len(fragments) if fragments else 0.0

    summary["similarity"].append(sim)
    summary["residue_acc"].append(residue_acc)
    summary["fragment_acc"].append(fragment_acc)

    print(f"Sample {i}")
    print(f"  Similarity: {sim:.4f}")
    print(f"  Correct Residues: {correct_residues}/{max_len} ({residue_acc:.4f})")
    print(
        f"  Correct Fragment Positions: {correctly_placed}/{len(fragments)} ({fragment_acc:.4f})"
    )

if samples:
    n = len(samples)
    avg = {k: sum(v) / n for k, v in summary.items()}
    model_name = cfg["mlm_model"]["name"]
    beam_size = cfg["mlm_model"].get("beam_size", "N/A")
    print("\nAverage Results")
    print("-" * 60)
    print(f"  Model: {model_name}")
    print(f"  Beam Size: {beam_size}")
    print(f"  Similarity: {avg['similarity']:.4f}")
    print(f"  Residue Accuracy: {avg['residue_acc']:.4f}")
    print(f"  Fragment Accuracy: {avg['fragment_acc']:.4f}")
