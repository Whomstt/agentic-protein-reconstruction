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


def _configure_model_profile(config: dict, section_name: str) -> None:
    section = config.get(section_name, {})
    profiles = section.get("profiles")
    selected_profile = section.get("profile")

    if not profiles:
        return

    if selected_profile not in profiles:
        valid = ", ".join(sorted(profiles))
        raise ValueError(
            f"Unknown {section_name}.profile '{selected_profile}'. Expected one of: {valid}"
        )

    resolved = {key: value for key, value in section.items() if key != "profiles"}
    profile_config = profiles[selected_profile]

    for key, value in profile_config.items():
        if key == "validity_model":
            continue
        resolved[key] = value

    resolved["profile"] = selected_profile
    config[section_name] = resolved

    if section_name == "llm_model":
        kind = resolved.get("kind")
        if kind == "microsoft_foundry":
            if "endpoint_env" not in resolved:
                raise ValueError(
                    f"llm_model.profile '{selected_profile}' must define endpoint_env"
                )
            if "token_scope" not in resolved:
                raise ValueError(
                    f"llm_model.profile '{selected_profile}' must define token_scope"
                )

    if section_name == "mlm_model":
        validity_model = profile_config.get("validity_model")
        if not validity_model:
            raise ValueError(
                f"mlm_model.profile '{selected_profile}' must define validity_model settings"
            )
        config["validity_model"] = dict(validity_model)


_configure_model_profile(cfg, "llm_model")
_configure_model_profile(cfg, "mlm_model")


_configure_active_dataset(cfg)
