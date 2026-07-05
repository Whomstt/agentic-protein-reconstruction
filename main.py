import json
import os
import textwrap

from agents.iterative_runner import run_iterative_reconstruction
from agents.react_agent import build_agent
from config import cfg
from dotenv import load_dotenv
from openai import AuthenticationError

load_dotenv(override=True)

DIM = "\033[2m"
BOLD = "\033[1m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
MAGENTA = "\033[35m"
RESET = "\033[0m"

WRAP_WIDTH = 96


def _wrap(text: str, indent: str) -> str:
    wrapper = textwrap.TextWrapper(
        width=WRAP_WIDTH,
        initial_indent=indent,
        subsequent_indent=indent,
        break_long_words=False,
        break_on_hyphens=False,
    )
    return "\n".join(wrapper.wrap(text)) if text else indent


def _format_args(args: dict) -> str:
    if not args:
        return ""
    parts = [f"{k}={v!r}" for k, v in args.items()]
    return ", ".join(parts)


def _preview(value, length=90) -> str:
    text = json.dumps(value) if isinstance(value, (dict, list)) else str(value)
    return text if len(text) <= length else text[:length] + "..."


def make_event_printer():
    """Returns an on_event(kind, payload) callback that live-prints agent reasoning."""

    def on_event(kind, payload):
        if kind == "iteration_start":
            i, n = payload["iteration"], payload["max_iterations"]
            print(f"\n{BOLD}{CYAN}┌─ Iteration {i}/{n} {'─' * 40}{RESET}")

        elif kind == "thought":
            print(f"{CYAN}│{RESET} {DIM}thinking:{RESET}")
            print(_wrap(payload, indent=f"{CYAN}│{RESET}   "))

        elif kind == "tool_call":
            name = payload["name"]
            args_str = _format_args(payload.get("args", {}))
            print(f"{CYAN}│{RESET} {MAGENTA}→ {name}{RESET}({DIM}{args_str}{RESET})")

        elif kind == "tool_result":
            name = payload["name"]
            content_preview = _preview(payload.get("content"))
            print(f"{CYAN}│{RESET}   {DIM}⤷ {content_preview}{RESET}")

        elif kind == "iteration_end":
            score = payload.get("validity_score")
            score_text = f"{score:.4f}" if isinstance(score, (int, float)) else "n/a"
            summary = payload.get("strategy_summary", "")
            print(
                f"{CYAN}└─{RESET} {BOLD}score={score_text}{RESET} {DIM}| {summary}{RESET}"
            )

    return on_event


def main():
    llm_config = cfg["llm_model"]
    api_key_env = llm_config.get("api_key_env", "OPENAI_API_KEY")
    api_key = os.environ.get(api_key_env, "").strip().strip("'\"")
    if llm_config.get("kind") == "microsoft_foundry":
        endpoint_env = llm_config["endpoint_env"]
        endpoint = os.environ.get(endpoint_env, "").strip().strip("'\"")
        if not endpoint:
            print(
                f"\n{YELLOW}Set {endpoint_env} in your environment or .env file.{RESET}"
            )
            return
        if llm_config.get("auth_mode", "auto") == "api_key" and not api_key:
            print(
                f"\n{YELLOW}Set {api_key_env} in your environment or .env file.{RESET}"
            )
            return
    elif not api_key:
        print(f"\n{YELLOW}Set {api_key_env} in your environment or .env file.{RESET}")
        return

    if api_key:
        os.environ[api_key_env] = api_key

    print(f"\n{BOLD}{'═' * 60}{RESET}")
    print(f"{BOLD}  Protein Reconstruction Agent{RESET}")
    print(
        f"{DIM}  Device: {cfg['misc']['device']}  │  "
        f"Dataset: {cfg['data']['organism_display_name']}  │  "
        f"MLM: {cfg['mlm_model']['name'].split('/')[-1]}  │  "
        f"LLM: {cfg['llm_model']['model']} ({cfg['llm_model']['profile']}){RESET}"
    )
    print(f"{BOLD}{'═' * 60}{RESET}")

    try:
        agent = build_agent()
    except AuthenticationError:
        print(
            f"\n{YELLOW}{api_key_env} was rejected by the selected LLM provider. Check that the key is current and copied exactly.{RESET}"
        )
        return

    with open(cfg["data"]["active_fragmented_split"]) as f:
        sample = json.loads(f.readline())

    fragment_samples = sample.get("fragment_samples")
    fragments = sample["fragments"]
    target = sample.get(
        cfg["data"]["active_target_key"], sample.get("target_reconstruction")
    )

    if fragment_samples:
        print(
            f"\n{BOLD}  Input: {len(fragment_samples)} digestion sample(s), {len(fragments)} unique fragments{RESET}"
        )
    else:
        print(f"\n{BOLD}  Input: {len(fragments)} fragments{RESET}")
    for i, frag in enumerate(fragments):
        preview = frag[:30] + "..." if len(frag) > 30 else frag
        print(f"{DIM}  [{i}] {preview} ({len(frag)} aa){RESET}")

    print(f"\n{BOLD}{'─' * 60}{RESET}")
    print(f"{BOLD}  Live reasoning{RESET}")
    print(f"{'─' * 60}")

    try:
        run_result = run_iterative_reconstruction(
            agent, fragments, fragment_samples, on_event=make_event_printer()
        )
    except AuthenticationError:
        print(
            f"\n{YELLOW}The selected LLM rejected the API key during the agent run. Update {api_key_env} and try again.{RESET}"
        )
        return

    best_record = run_result.get("best_record", {})
    iteration_history = run_result.get("iteration_history", [])
    reconstruction = best_record.get("reconstruction", "")
    validity_score = best_record.get("validity_score")

    print(f"\n{'─' * 60}")
    print(f"{BOLD}  Iteration overview{RESET}")
    print(f"{'─' * 60}")
    for record in iteration_history:
        score = record.get("validity_score")
        summary = record.get("strategy_summary", "")
        score_text = f"{score:.4f}" if isinstance(score, (int, float)) else "n/a"
        marker = " *" if record is best_record else "  "
        print(
            f"{marker}Iteration {record['iteration']}: score={score_text} | {summary}"
        )

    print(f"\n{'═' * 60}")
    print(f"{BOLD}  Result{RESET}")
    print(f"{'─' * 60}")

    if reconstruction:
        match = target == reconstruction
        status = (
            f"{GREEN}✓ Exact match{RESET}" if match else f"{YELLOW}✗ Mismatch{RESET}"
        )
        print(
            f"  Target:        {DIM}{target[:70]}{'...' if len(target) > 70 else ''}{RESET}"
        )
        print(
            f"  Reconstructed: {DIM}{reconstruction[:70]}{'...' if len(reconstruction) > 70 else ''}{RESET}"
        )
        if isinstance(validity_score, (int, float)):
            print(f"  Validity score: {DIM}{validity_score:.4f}{RESET}")
        print(f"  {status}")
    else:
        print(f"  {YELLOW}No reconstruction produced{RESET}")

    print(f"{'═' * 60}\n")


if __name__ == "__main__":
    main()
