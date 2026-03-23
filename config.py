import yaml
import torch

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

if cfg["misc"]["device"] == "auto":
    cfg["misc"]["device"] = "cuda" if torch.cuda.is_available() else "cpu"
