import os
import random
from pathlib import Path

import numpy as np
import yaml
import torch

# Sweeps (see evaluation/sweep.py) point this at a generated per-combo
# config file so each subprocess run reads its own overrides without ever
# touching the checked-in config.yaml.
CONFIG_PATH = Path(
    os.environ.get("AGENTIC_CONFIG_PATH")
    or Path(__file__).resolve().parent / "config.yaml"
)


def _load_dotenv(path: Path) -> None:
    """Minimal, dependency-free .env loader so `python main.py` picks up API
    keys regardless of launcher (a plain terminal doesn't inject .env the way
    the VSCode Python extension does). Real environment variables always win —
    values are only filled in when not already set — so shell exports, CI, and
    the sweep parent's env are never clobbered."""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


# Anchor .env to the project root, not CONFIG_PATH's dir (which points at a
# temp per-combo config during sweeps).
_load_dotenv(Path(__file__).resolve().parent / ".env")

with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

if cfg["misc"]["device"] == "auto":
    cfg["misc"]["device"] = "cuda" if torch.cuda.is_available() else "cpu"


def _seed_everything(seed: int) -> None:
    """Single source of truth for randomness. Every module that needs a
    random draw should use the plain global random/numpy/torch APIs (not its
    own seeded instance) so misc.seed fully determines the run."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


_seed_everything(cfg["misc"]["seed"])


def _configure_active_dataset(config: dict) -> None:
    dataset_name = config["data"].get("organism", "ecoli")
    dataset_map = {
        "ecoli": {
            "display_name": "E. coli",
            "fragmented_split": config["data"]["fragmented_ecoli"],
            "target_key": "ecoli_original",
        },
        "yeast": {
            "display_name": "Yeast",
            "fragmented_split": config["data"]["fragmented_yeast"],
            "target_key": "yeast_original",
        },
        "mixture": {
            "display_name": "Mixture",
            "fragmented_split": config["data"].get("fragmented_mixture"),
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
