"""Utility functions and helpers."""

import json
import random
from pathlib import Path
from typing import Dict, List
import numpy as np
import torch  # type: ignore
import yaml  # type: ignore


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> Dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def merge_configs(base_config: Dict, override_config: Dict) -> Dict:
    """Merge two configuration dictionaries."""
    merged = base_config.copy()

    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value

    return merged


def save_json(data: Dict, filepath: str):
    """Save dictionary as JSON."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)


def load_json(filepath: str) -> Dict:
    """Load JSON file."""
    with open(filepath, "r") as f:
        return json.load(f)


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_device() -> torch.device:
    """Get the appropriate device (GPU if available, else CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_item_list(filepath: str) -> List[str]:
    """Load list of item IDs from text file."""
    with open(filepath, "r") as f:
        return [line.strip() for line in f if line.strip()]


def save_embeddings(embeddings: Dict[str, np.ndarray], filepath: str):  # type: ignore[misc]
    """Save item embeddings to file."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    np.savez(filepath, **embeddings)  # type: ignore[arg-type]


def load_embeddings(filepath: str) -> Dict[str, np.ndarray]:
    """Load item embeddings from file."""
    data = np.load(filepath)
    return {key: data[key] for key in data.files}
