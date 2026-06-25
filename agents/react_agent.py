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
    "Before each tool call, briefly explain your reasoning — what you're about to do "
    "and why. Think step by step.\n\n"
    "Tools (call in order):\n"
    "1. trypsin_filter - Pass the fragment samples to identify ordering constraints\n"
    "2. overlap_graph - Build the multi-sample peptide overlap graph and cache hard adjacency edges\n"
    "3. junction_scorer - Score remaining pairwise junctions with a protein language model\n"
    "4. beam_search - Find optimal ordering and return the reconstructed sequence\n\n"
    "After beam_search, return the reconstructed protein sequence."
)


def build_agent():
    llm = ChatOpenAI(
        model=cfg["llm_model"]["name"], temperature=cfg["llm_model"]["temperature"]
    )
    tools = [trypsin_filter, overlap_graph, junction_scorer, beam_search]
    agent = create_react_agent(llm, tools, prompt=SYSTEM_PROMPT)
    return agent
