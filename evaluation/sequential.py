import json
from difflib import SequenceMatcher
from tools.reconstruction_tool import reconstruct_tool
from config import cfg


test_path = cfg["data"]["ecoli_test_split"]
sample_count = 3  # number of samples to evaluate (None to evaluate all)

with open(test_path) as f:
    samples = [json.loads(l) for l in f if l.strip()][:sample_count]


def all_occurrences(text, pattern):
    i, starts = text.find(pattern), []
    while i != -1:
        starts.append(i)
        i = text.find(pattern, i + 1)
    return starts


def count_correctly_placed_fragments(original, fragments, order):
    used, correct, placed, cursor = set(), 0, 0, 0
    for idx in order:
        frag = fragments[idx]
        true_starts = all_occurrences(original, frag)
        placed += bool(true_starts)
        if cursor in true_starts and cursor not in used:
            correct += 1
            used.add(cursor)
        cursor += len(frag)
    return correct, placed


summary = {
    "similarity": [],
    "exact_match": [],
    "len_delta": [],
    "residue_acc": [],
    "fragments_present_rate": [],
    "fragments_correctly_placed_rate": [],
}

print(f"Sequential Reconstruction Evaluation ({len(samples)} Samples)")
print("-" * 60)


for i, sample in enumerate(samples, 1):
    original_key = next(k for k in sample if k.endswith("_original"))
    original = sample[original_key]
    fragments = sample["fragments"]

    result = reconstruct_tool.invoke({"fragments": fragments})
    reconstruction, order = result["reconstruction"], result["order"]

    sim = SequenceMatcher(None, original, reconstruction).ratio()
    residue_matches = sum(a == b for a, b in zip(original, reconstruction))
    residue_acc = residue_matches / len(original) if original else 0.0
    correctly_placed, fragments_present = count_correctly_placed_fragments(
        original, fragments, order
    )
    len_delta = len(reconstruction) - len(original)
    fragments_present_rate = fragments_present / len(fragments) if fragments else 0.0
    correctly_placed_rate = correctly_placed / len(fragments) if fragments else 0.0

    summary["similarity"].append(sim)
    summary["exact_match"].append(float(reconstruction == original))
    summary["len_delta"].append(len_delta)
    summary["residue_acc"].append(residue_acc)
    summary["fragments_present_rate"].append(fragments_present_rate)
    summary["fragments_correctly_placed_rate"].append(correctly_placed_rate)

    print(f"Sample {i}")
    print(
        f"  Similarity: {sim:.4f} | Exact Match: {"Yes" if reconstruction == original else "No"} | Length Delta: {len_delta:+d}"
    )
    print(
        f"  Residues Correct: {residue_matches}/{len(original)} ({residue_acc:.4f}) | Fragments Present: {fragments_present}/{len(fragments)} | Correctly Placed Fragments: {correctly_placed}/{len(fragments)}"
    )

if samples:
    n = len(samples)
    avg = {k: sum(v) / n for k, v in summary.items()}
    print("\nAverage Results")
    print("-" * 60)
    print(
        f"Similarity: {avg["similarity"]:.4f} | Exact Match Rate: {avg["exact_match"]:.4f} | Avg Length Delta: {avg["len_delta"]:+.2f}"
    )
    print(
        f"Residue Accuracy: {avg["residue_acc"]:.4f} | Fragments Present Rate: {avg["fragments_present_rate"]:.4f} | Correctly Placed Fragments Rate: {avg["fragments_correctly_placed_rate"]:.4f}"
    )
