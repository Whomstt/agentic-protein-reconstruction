from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from tools.trypsin_filter import trypsin_filter
from tools.junction_scorer import junction_scorer
from tools.beam_search import beam_search
from config import cfg

SYSTEM_PROMPT = (
    "You are a protein reconstruction agent. Given unordered protein fragments from "
    "trypsin digestion, reconstruct the original protein sequence.\n\n"
    "Before each tool call, briefly explain your reasoning — what you're about to do "
    "and why. Think step by step.\n\n"
    "Tools (call in order):\n"
    "1. trypsin_filter - Pass the fragments to identify ordering constraints\n"
    "2. junction_scorer - Score pairwise junctions with a protein language model\n"
    "3. beam_search - Find optimal ordering and return the reconstructed sequence\n\n"
    "After beam_search, return the reconstructed protein sequence."
)


def build_agent():
    llm = ChatOpenAI(
        model=cfg["llm_model"]["name"], temperature=cfg["llm_model"]["temperature"]
    )
    tools = [trypsin_filter, junction_scorer, beam_search]
    agent = create_react_agent(llm, tools, prompt=SYSTEM_PROMPT)
    return agent
