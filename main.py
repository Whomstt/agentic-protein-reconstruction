import json
from agents.react_agent import build_agent
from config import cfg

DIM = "\033[2m"
BOLD = "\033[1m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RESET = "\033[0m"

TOOL_ICONS = {
    "trypsin_filter": "🔬",
    "junction_scorer": "🧬",
    "beam_search": "🔍",
}


def fmt_tool_result(name, content):
    """Summarize tool output concisely."""
    try:
        data = json.loads(content) if isinstance(content, str) else content
    except (json.JSONDecodeError, TypeError):
        return str(content)

    if name == "trypsin_filter":
        return data.get("message", str(data))
    elif name == "junction_scorer":
        n = data.get("num_fragments", "?")
        return f"{n}x{n} junction score matrix computed"
    elif name == "beam_search":
        order = data.get("order", [])
        seq = data.get("reconstruction", "")
        return f"Order: {order} ({len(seq)} residues)"
    return str(data)


def print_event(event):
    if "agent" in event:
        for msg in event["agent"].get("messages", []):
            # Agent thinking / reasoning
            if hasattr(msg, "content") and msg.content:
                print(f"\n  {YELLOW}💭 {msg.content}{RESET}")

            # Tool calls
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    name = tc["name"]
                    icon = TOOL_ICONS.get(name, "⚙")
                    args = tc.get("args", {})
                    if args and "fragments" in args:
                        arg_str = f"[{len(args['fragments'])} fragments]"
                    elif args:
                        arg_str = ", ".join(f"{k}={v}" for k, v in args.items())
                    else:
                        arg_str = ""
                    print(f"  {CYAN}{icon} {BOLD}{name}{RESET}{CYAN}({arg_str}){RESET}")

    if "tools" in event:
        for msg in event["tools"].get("messages", []):
            name = msg.name if hasattr(msg, "name") else "tool"
            content = msg.content if hasattr(msg, "content") else str(msg)
            summary = fmt_tool_result(name, content)
            print(f"  {GREEN}   ✓ {summary}{RESET}")


def main():
    print(f"\n{BOLD}{'═' * 60}{RESET}")
    print(f"{BOLD}  Protein Reconstruction Agent{RESET}")
    print(f"{DIM}  Device: {cfg['misc']['device']}  │  "
          f"MLM: {cfg['mlm_model']['name'].split('/')[-1]}  │  "
          f"LLM: {cfg['llm_model']['name']}{RESET}")
    print(f"{BOLD}{'═' * 60}{RESET}")

    agent = build_agent()

    with open(cfg["data"]["fragmented_ecoli"]) as f:
        sample = json.loads(f.readline())

    fragments = sample["fragments"]
    original = sample["ecoli_original"]

    print(f"\n{BOLD}  Input: {len(fragments)} fragments{RESET}")
    for i, frag in enumerate(fragments):
        preview = frag[:30] + "..." if len(frag) > 30 else frag
        print(f"{DIM}  [{i}] {preview} ({len(frag)} aa){RESET}")

    print(f"\n{'─' * 60}")

    reconstruction = None
    for event in agent.stream(
        {"messages": [("user", f"Reconstruct the protein from these fragments: {fragments}")]},
        stream_mode="updates",
    ):
        print_event(event)

        # Capture reconstruction from beam_search result
        if "tools" in event:
            for msg in event["tools"].get("messages", []):
                if hasattr(msg, "name") and msg.name == "beam_search":
                    content = msg.content if hasattr(msg, "content") else str(msg)
                    try:
                        data = json.loads(content) if isinstance(content, str) else content
                        reconstruction = data.get("reconstruction", "")
                    except (json.JSONDecodeError, TypeError):
                        pass

    print(f"\n{'═' * 60}")
    print(f"{BOLD}  Result{RESET}")
    print(f"{'─' * 60}")

    if reconstruction:
        match = original == reconstruction
        status = f"{GREEN}✓ Exact match{RESET}" if match else f"{YELLOW}✗ Mismatch{RESET}"
        print(f"  Original:      {DIM}{original[:70]}{'...' if len(original) > 70 else ''}{RESET}")
        print(f"  Reconstructed: {DIM}{reconstruction[:70]}{'...' if len(reconstruction) > 70 else ''}{RESET}")
        print(f"  {status}")
    else:
        print(f"  {YELLOW}No reconstruction produced{RESET}")

    print(f"{'═' * 60}\n")


if __name__ == "__main__":
    main()
