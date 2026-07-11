import os
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from langchain_openai import ChatOpenAI, AzureChatOpenAI
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
    f"A fragment sample is already loaded in shared state. You will be called for up to {cfg['search']['max_iterations']} iteration(s). Each iteration must test a materially different reconstruction strategy based on what failed before. The only controllable strategy levers are: 1) junction masking window via junction_scorer(window=...) or beam_search(window=...), 2) search mode via beam_search(search_mode='beam'|'greedy'), 3) beam width via beam_search(beam_width=...), 4) edge mode via beam_search(edge_mode='hard'|'soft'), and 5) confirmed-edge bonus via beam_search(confirmed_bonus=...). Do not treat any other knob as a strategy lever.\n\n"
    f"Tie each lever to a failure signal. If junction scores look weak, broaden or narrow the masking window and, when only a few pairs are suspect, rescore just those junction pairs instead of recomputing everything. If the search cuts off early or collapses to a partial path, change search_mode or adjust beam_width by roughly {cfg['search']['beam_width_step']} at a time rather than a large jump. If the overlap graph and MLM disagree, switch edge_mode or adjust confirmed_bonus. Do not simply increase beam width monotonically.\n\n"
    f"After each candidate reconstruction, call validity_scorer on that reconstruction. Lower pseudo-perplexity is better, and the run stops early once the best validity score fails to improve for {cfg['search']['early_stop_patience']} consecutive iterations, otherwise it continues until iteration {cfg['search']['max_iterations']}. Store the reconstruction, lever choices, and validity score in shared state through the tools. At the end of the allotted iterations, return the best scoring reconstruction found.\n\n"
    "Use the fewest calls necessary within an iteration, and keep text responses brief. On the first iteration, start with trypsin_filter to initialize constraints, then overlap_graph, junction_scorer, beam_search, and validity_scorer. Constraints and the overlap graph do not change between iterations, so on later iterations skip trypsin_filter and overlap_graph and go straight to junction_scorer/beam_search with your new lever choices.\n\n"
    "Available tools:\n"
    "- trypsin_filter: initialize constraints from the preloaded fragment sample or an explicit fragment list\n"
    "- overlap_graph: build the peptide overlap graph and cache hard adjacency edges\n"
    "- junction_scorer: score remaining pairwise junctions with a protein language model; pass a custom window to retest a different local context, or pass a subset of junction pairs to rescore and merge into the existing matrix\n"
    "- beam_search: find the final ordering and reconstructed sequence; pass search_mode, beam_width, edge_mode, confirmed_bonus, and window to vary only the five allowed strategy levers\n"
    "- validity_scorer: compute an ESM-2 pseudo-perplexity validity score for a candidate reconstruction\n\n"
    "After the final validity check in an iteration, return the reconstruction that you believe should be retained for this round."
)


def build_agent():
    llm_config = cfg["llm_model"]
    api_key_env = llm_config.get("api_key_env", "OPENAI_API_KEY")
    api_key = os.environ.get(api_key_env, "").strip()

    if llm_config.get("kind") == "microsoft_foundry":
        endpoint = os.environ.get(llm_config["endpoint_env"], "").strip()
        # Ensure the endpoint points to the v1 path
        if not endpoint.endswith("/openai/v1"):
            endpoint = f"{endpoint.rstrip('/')}/openai/v1"

        auth_mode = llm_config.get("auth_mode", "auto")
        final_key = (
            api_key
            if (auth_mode == "api_key" or (auth_mode == "auto" and api_key))
            else get_bearer_token_provider(
                DefaultAzureCredential(), llm_config["token_scope"]
            )()
        )

        # Use standard ChatOpenAI for Foundry-hosted models
        llm = ChatOpenAI(
            model=llm_config["model"],
            base_url=endpoint,
            api_key=final_key,
            default_query={
                "api-version": os.environ.get(
                    "AZURE_OPENAI_API_VERSION", "2025-11-15-preview"
                )
            },
            temperature=llm_config.get("temperature", 0.0),
        )
    else:
        if not api_key:
            raise ValueError(f"{api_key_env} is not set.")
        llm = ChatOpenAI(
            model=llm_config["model"],
            api_key=api_key,
            temperature=llm_config.get("temperature", 0.0),
        )

    tools = [
        trypsin_filter,
        overlap_graph,
        junction_scorer,
        beam_search,
        validity_scorer,
    ]
    return create_react_agent(llm, tools, prompt=SYSTEM_PROMPT)
