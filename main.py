import json
from agents.react_agent import build_agent
from config import cfg


def main():
    """Main function to run the agent on a sample protein reconstruction task."""

    print(f"Using device: {cfg["misc"]["device"]}")

    agent = build_agent()

    with open(cfg["data"]["fragmented_ecoli"]) as f:
        sample = json.loads(f.readline())

    fragments = sample["fragments"]
    original = sample["ecoli_original"]

    result = agent.invoke(
        {
            "messages": [
                (
                    "user",
                    f"Reconstruct the protein sequence from these fragments: {fragments}",
                )
            ]
        }
    )

    print(f"Original:      {original}")
    print(f"Reconstructed: {result['messages'][-1].content}")


if __name__ == "__main__":
    main()
