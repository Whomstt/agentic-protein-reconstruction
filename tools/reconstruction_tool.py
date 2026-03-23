from langchain.tools import tool
from algorithms.score_junctions import score_junctions
from algorithms.greedy_order import greedy_order
from algorithms.beam_order import beam_order


@tool
def reconstruct_tool(fragments: list[str]) -> dict:
    """Reconstruct a protein sequence from unordered fragments using ProtBERT MLM scores."""
    scores = score_junctions(fragments)
    order = beam_order(scores)  # greedy or beam ordering
    result = "".join(fragments[i] for i in order)
    return {
        "reconstruction": result,  # output reconstructed sequence
        "order": order,  # order of fragments used in reconstruction
        "scores": scores.tolist(),  # pairwise junction scores for reference
    }
