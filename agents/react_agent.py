from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from tools.trypsin_filter import trypsin_filter
from tools.overlap_graph import overlap_graph
from tools.junction_scorer import junction_scorer
from tools.beam_search import beam_search
from config import cfg

SYSTEM_PROMPT = (
    "You are a protein reconstruction agent. Given unordered protein fragments from "
    "trypsin digestion, reconstruct the original protein sequence.\n\n"
    "A fragment sample is already loaded in shared state. Decide which tools to use, "
    "use the fewest calls necessary, and keep text responses brief. In most cases, "
    "start with trypsin_filter to initialize constraints, then use overlap_graph, "
    "junction_scorer, and beam_search as needed.\n\n"
    "Available tools:\n"
    "- trypsin_filter: initialize constraints from the preloaded fragment sample or an explicit fragment list\n"
    "- overlap_graph: build the peptide overlap graph and cache hard adjacency edges\n"
    "- junction_scorer: score remaining pairwise junctions with a protein language model\n"
    "- beam_search: find the final ordering and reconstructed sequence\n\n"
    "After beam_search, return the reconstructed protein sequence."
)


def build_agent():
    llm = ChatOpenAI(
        model=cfg["llm_model"]["name"], temperature=cfg["llm_model"]["temperature"]
    )
    tools = [trypsin_filter, overlap_graph, junction_scorer, beam_search]
    agent = create_react_agent(llm, tools, prompt=SYSTEM_PROMPT)
    return agent
