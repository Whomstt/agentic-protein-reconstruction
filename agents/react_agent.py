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
    "Every tool returns diagnostics beyond a bare pass/fail — read them before picking the next iteration's levers, don't guess blind:\n"
    "- junction_scorer returns mean_score/min_score/max_score over the pairs it just scored. A narrow spread near the mean means the current junction_window isn't giving the PLM enough signal to discriminate real junctions from wrong ones — try a different window (narrower for very local motifs, wider for more successor-fragment context).\n"
    "- beam_search returns fell_back (search_mode='beam': true means beam_width/edge_mode cut the search off before covering every fragment — widen beam_width or try edge_mode='soft'), forced_impossible_count (search_mode='greedy': non-zero means trypsin constraints fought the ordering at some step — try search_mode='beam' instead), num_confirmed_edges_realized/num_confirmed_edges_total (how many overlap-confirmed adjacencies made it into the final order), and mean_junction_score (a cheap preview of ordering quality before paying for validity_scorer).\n"
    "- validity_scorer returns validity_score (lower is better; this is what decides which iteration's candidate wins) plus its two components: junction_local_ppl (PLM implausibility of the ordering's non-confirmed junctions — high means change junction_window) and confirmed_adjacency_agreement (fraction of real overlap-confirmed adjacencies the ordering respected — low means switch edge_mode to 'hard' or raise confirmed_bonus in 'soft' mode). Use these two numbers, not just the blended score, to diagnose *why* a candidate scored the way it did.\n\n"
    "Tie each lever to the failure signal above, not to a hunch. If the search cuts off early or collapses to a partial path (fell_back or forced_impossible_count), change search_mode or beam_width — you decide how much to change it, sized to how far the search fell short, rather than making an erratic large jump. Do not simply increase beam width monotonically.\n\n"
    f"After each candidate reconstruction, call validity_scorer on that reconstruction. The run stops early once the best validity_score fails to improve for {cfg['search']['early_stop_patience']} consecutive iterations, otherwise it continues until iteration {cfg['search']['max_iterations']}. Store the reconstruction, lever choices, and validity score in shared state through the tools. At the end of the allotted iterations, return the best scoring reconstruction found.\n\n"
    "Use the fewest calls necessary within an iteration, and keep text responses brief. On the first iteration, start with trypsin_filter to initialize constraints, then overlap_graph, junction_scorer, beam_search, and validity_scorer. Constraints and the overlap graph do not change between iterations, so on later iterations skip trypsin_filter and overlap_graph and go straight to junction_scorer/beam_search with your new lever choices.\n\n"
    "Available tools:\n"
    "- trypsin_filter: initialize constraints from the preloaded fragment sample or an explicit fragment list; reports how many junctions were pruned as chemically impossible (K/R->P, non-K/R-terminal) and how many N-terminal start candidates exist\n"
    "- overlap_graph: build the peptide overlap graph from multi-replica digestions and cache hard adjacency edges confirmed by real overlap evidence, not model guesswork\n"
    "- junction_scorer: score remaining pairwise junctions with a protein language model; pass a custom window to retest a different local context, or pass a subset of junction_pairs to rescore and merge into the existing matrix once you know which junctions are weak\n"
    "- beam_search: find the final ordering and reconstructed sequence; pass search_mode, beam_width, edge_mode, confirmed_bonus, and window to vary only the five allowed strategy levers; returns search diagnostics (see above)\n"
    "- validity_scorer: compute the blended junction+overlap validity score plus its two components (see above) for a candidate reconstruction\n\n"
    "After the final validity check in an iteration, return the reconstruction that you believe should be retained for this round."
)


class Agent:
    """Mode-tagged wrapper so iterative_runner can dispatch on how the LLM is used.

    mode="react": .graph is a compiled LangGraph ReAct agent that drives every
    tool call itself. mode="single_call": .llm is the raw chat model; the
    harness runs the fixed tool pipeline deterministically and only asks the
    LLM to choose the five strategy levers, once per iteration.
    """

    def __init__(self, mode: str, llm=None, graph=None):
        self.mode = mode
        self.llm = llm
        self.graph = graph


def _llm_sampling_kwargs() -> dict:
    """Translate llm_model.sampling into ChatOpenAI kwargs, sending only the
    knobs the user actually set. Reasoning models (gpt-5*/o-series) reject any
    temperature/top_p other than the default, so those are omitted unless the
    config explicitly sets them; reasoning_effort/verbosity/seed are the levers
    that do apply. top_k is not an OpenAI parameter, so it is routed through
    model_kwargs only when set (a no-op for OpenAI/Azure, honored by backends
    that support it)."""
    sampling = cfg["llm_model"].get("sampling") or {}
    kwargs: dict = {}
    for key in ("temperature", "top_p", "seed", "reasoning_effort", "verbosity"):
        value = sampling.get(key)
        if value is not None:
            kwargs[key] = value

    top_k = sampling.get("top_k")
    if top_k is not None:
        kwargs["model_kwargs"] = {"top_k": top_k}

    return kwargs


def build_llm():
    llm_config = cfg["llm_model"]
    api_key_env = llm_config.get("api_key_env", "OPENAI_API_KEY")
    api_key = os.environ.get(api_key_env, "").strip()
    sampling_kwargs = _llm_sampling_kwargs()

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
        return ChatOpenAI(
            model=llm_config["model"],
            base_url=endpoint,
            api_key=final_key,
            default_query={
                "api-version": os.environ.get(
                    "AZURE_OPENAI_API_VERSION", "2025-11-15-preview"
                )
            },
            **sampling_kwargs,
        )

    if not api_key:
        raise ValueError(f"{api_key_env} is not set.")
    return ChatOpenAI(
        model=llm_config["model"],
        api_key=api_key,
        **sampling_kwargs,
    )


def build_agent() -> Agent:
    llm = build_llm()
    calling_mode = cfg["run"].get("calling_mode", "react")

    if calling_mode == "single_call":
        return Agent(mode="single_call", llm=llm)

    tools = [
        trypsin_filter,
        overlap_graph,
        junction_scorer,
        beam_search,
        validity_scorer,
    ]
    graph = create_react_agent(llm, tools, prompt=SYSTEM_PROMPT)
    return Agent(mode="react", graph=graph)
