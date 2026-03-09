import json
from agents.react_agent import build_agent


def main():
    agent = build_agent()

    with open("data/fragmented_ecoli.jsonl") as f:
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
