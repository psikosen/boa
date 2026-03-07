"""Utility helpers for Bash-MANTIS."""

from __future__ import annotations

import random
import torch
import numpy as np


def count_parameters(model: torch.nn.Module, trainable_only: bool = True) -> int:
    """Count parameters in a model."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def count_parameters_by_group(model: torch.nn.Module) -> dict[str, int]:
    """Count parameters grouped by top-level module name."""
    groups: dict[str, int] = {}
    for name, param in model.named_parameters():
        group = name.split(".")[0]
        groups[group] = groups.get(group, 0) + param.numel()
    return groups


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
