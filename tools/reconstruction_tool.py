from langchain.tools import tool
from algorithms.score_junctions import score_junctions
from algorithms.greedy_order import greedy_order


@tool
def reconstruct_protein_fragments(fragments: list[str]) -> dict:
    """Reconstruct a protein sequence from unordered fragments using ProtBERT MLM scores."""
    scores = score_junctions(fragments)
    order = greedy_order(scores)
    result = "".join(fragments[i] for i in order)
    return {
        "order": order,
        "reconstruction": result,
        "scores": scores.tolist(),
    }
