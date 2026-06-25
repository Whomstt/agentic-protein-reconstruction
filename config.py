from pathlib import Path

import yaml
import torch

with open(Path(__file__).resolve().parent / "config.yaml") as f:
    cfg = yaml.safe_load(f)

if cfg["misc"]["device"] == "auto":
    cfg["misc"]["device"] = "cuda" if torch.cuda.is_available() else "cpu"


def _configure_active_dataset(config: dict) -> None:
    dataset_name = config["data"].get("organism", "ecoli")
    dataset_map = {
        "ecoli": {
            "display_name": "E. coli",
            "fragmented_split": config["data"]["fragmented_ecoli"],
            "test_split": config["data"]["ecoli_test_split"],
            "train_split": config["data"].get("ecoli_train_split"),
            "target_key": "ecoli_original",
        },
        "yeast": {
            "display_name": "Yeast",
            "fragmented_split": config["data"]["fragmented_yeast"],
            "test_split": config["data"]["yeast_test_split"],
            "train_split": config["data"].get("yeast_train_split"),
            "target_key": "yeast_original",
        },
        "mixture": {
            "display_name": "Mixture",
            "fragmented_split": config["data"].get("fragmented_mixture"),
            "test_split": config["data"].get("mixture_test_split"),
            "train_split": config["data"].get("mixture_train_split"),
            "target_key": "target_reconstruction",
        },
    }

    if dataset_name not in dataset_map:
        valid = ", ".join(sorted(dataset_map))
        raise ValueError(
            f"Unknown data.organism '{dataset_name}'. Expected one of: {valid}"
        )

    active = dataset_map[dataset_name]
    config["data"]["organism"] = dataset_name
    config["data"]["organism_display_name"] = active["display_name"]
    config["data"]["active_fragmented_split"] = active["fragmented_split"]
    config["data"]["active_test_split"] = active["test_split"]
    config["data"]["active_train_split"] = active["train_split"]
    config["data"]["active_target_key"] = active["target_key"]


_configure_active_dataset(cfg)
