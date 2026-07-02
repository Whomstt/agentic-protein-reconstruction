import json
import os

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
RESET = "\033[0m"


def main():
    api_key = os.environ.get("OPENAI_API_KEY", "").strip().strip("'\"")
    if not api_key:
        print(
            f"\n{YELLOW}OPENAI_API_KEY is not set in your environment or .env file.{RESET}"
        )
        return

    os.environ["OPENAI_API_KEY"] = api_key

    print(f"\n{BOLD}{'═' * 60}{RESET}")
    print(f"{BOLD}  Protein Reconstruction Agent{RESET}")
    print(
        f"{DIM}  Device: {cfg['misc']['device']}  │  "
        f"Dataset: {cfg['data']['organism_display_name']}  │  "
        f"MLM: {cfg['mlm_model']['name'].split('/')[-1]}  │  "
        f"LLM: {cfg['llm_model']['name']}{RESET}"
    )
    print(f"{BOLD}{'═' * 60}{RESET}")

    try:
        agent = build_agent()
    except AuthenticationError:
        print(
            f"\n{YELLOW}OPENAI_API_KEY was rejected by OpenAI. Check that the key is current and copied exactly.{RESET}"
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

    try:
        run_result = run_iterative_reconstruction(agent, fragments, fragment_samples)
    except AuthenticationError:
        print(
            f"\n{YELLOW}OpenAI rejected the API key during the agent run. Update OPENAI_API_KEY and try again.{RESET}"
        )
        return

    best_record = run_result.get("best_record", {})
    iteration_history = run_result.get("iteration_history", [])
    reconstruction = best_record.get("reconstruction", "")
    validity_score = best_record.get("validity_score")

    print(f"\n{'─' * 60}")
    print(f"{BOLD}  Iterations{RESET}")
    print(f"{'─' * 60}")
    for record in iteration_history:
        score = record.get("validity_score")
        summary = record.get("strategy_summary", "")
        score_text = f"{score:.4f}" if isinstance(score, (int, float)) else "n/a"
        print(f"  Iteration {record['iteration']}: score={score_text} | {summary}")

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
