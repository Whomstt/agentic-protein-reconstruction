from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from tools.trypsin_filter import trypsin_filter
from tools.overlap_graph import overlap_graph
from tools.junction_scorer import junction_scorer
from tools.beam_search import beam_search
from tools.validity_scorer import validity_scorer
from config import cfg

SYSTEM_PROMPT = (
    "You are a protein reconstruction agent. Given unordered protein fragments from "
    "trypsin digestion, reconstruct the original protein sequence.\n\n"
    f"A fragment sample is already loaded in shared state. You will be called for up to {cfg['search']['max_iterations']} iteration(s). Each iteration must test a materially different reconstruction strategy based on what failed before. Do not simply increase beam width monotonically. Use the previous validity score and reconstruction error to reason about why the last attempt was weak, then change the hypothesis: rerun junction_scorer with a different masking window, change beam_search between beam and greedy modes, adjust beam width by {cfg['search']['beam_width_step']}, or reweight confirmed versus unconfirmed overlap edges with edge_mode='soft' and a bonus. Keep the strategy targeted and different each round.\n\n"
    f"After each candidate reconstruction, call validity_scorer on that reconstruction. Lower pseudo-perplexity is better, and the run should stop early only if the validity score reaches {cfg['search']['validity_threshold']}. Store the reconstruction, strategy choice, and validity score in shared state through the tools. At the end of the allotted iterations, return the best scoring reconstruction found.\n\n"
    "Use the fewest calls necessary within an iteration, and keep text responses brief. In most cases, start with trypsin_filter to initialize constraints, then use overlap_graph, junction_scorer, beam_search, and validity_scorer as needed.\n\n"
    "Available tools:\n"
    "- trypsin_filter: initialize constraints from the preloaded fragment sample or an explicit fragment list\n"
    "- overlap_graph: build the peptide overlap graph and cache hard adjacency edges\n"
    "- junction_scorer: score remaining pairwise junctions with a protein language model; pass a custom window to retest a different local context\n"
    "- beam_search: find the final ordering and reconstructed sequence; pass search_mode, beam_width, and edge_mode to vary the strategy\n"
    "- validity_scorer: compute an ESM-2 pseudo-perplexity validity score for a candidate reconstruction\n\n"
    "After the final validity check in an iteration, return the reconstruction that you believe should be retained for this round."
)


def build_agent():
    llm = ChatOpenAI(
        model=cfg["llm_model"]["name"], temperature=cfg["llm_model"]["temperature"]
    )
    tools = [
        trypsin_filter,
        overlap_graph,
        junction_scorer,
        beam_search,
        validity_scorer,
    ]
    agent = create_react_agent(llm, tools, prompt=SYSTEM_PROMPT)
    return agent
