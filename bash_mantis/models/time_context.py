"""Temporal context conditioning for Bash-MANTIS.

Injects a small learned representation of the current time into the model,
allowing it to ground time-relative bash constructs (cron, find -mtime,
date -d, at/batch, etc.) without architectural changes to the backbone.

The encoding covers:
  - hour of day     (0-23)  → cyclic sin/cos
  - day of week     (0-6)   → cyclic sin/cos
  - day of month    (1-31)  → cyclic sin/cos
  - month           (1-12)  → cyclic sin/cos
  - is_weekend      (bool)  → scalar

The 9-dim raw feature vector is projected to d_model and added to the
token embeddings (broadcast across sequence length).
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn


# Number of raw cyclic + scalar features
_RAW_DIM = 9  # 4 pairs of sin/cos + 1 boolean


class TimeContext(nn.Module):
    """Encode wall-clock time as a conditioning bias on token embeddings."""

    def __init__(self, d_model: int = 108):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(_RAW_DIM, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def encode_time(
        hour: int,
        day_of_week: int,
        day_of_month: int,
        month: int,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Build the raw 9-dim feature vector for a single timestamp.

        Returns:
            [9] tensor
        """
        feats = [
            math.sin(2 * math.pi * hour / 24),
            math.cos(2 * math.pi * hour / 24),
            math.sin(2 * math.pi * day_of_week / 7),
            math.cos(2 * math.pi * day_of_week / 7),
            math.sin(2 * math.pi * day_of_month / 31),
            math.cos(2 * math.pi * day_of_month / 31),
            math.sin(2 * math.pi * month / 12),
            math.cos(2 * math.pi * month / 12),
            float(day_of_week >= 5),  # is_weekend
        ]
        return torch.tensor(feats, device=device)

    @staticmethod
    def encode_time_batch(
        hours: list[int],
        days_of_week: list[int],
        days_of_month: list[int],
        months: list[int],
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Batch version of encode_time.

        Returns:
            [B, 9] tensor
        """
        rows = [
            TimeContext.encode_time(h, dw, dm, m, device=device)
            for h, dw, dm, m in zip(hours, days_of_week, days_of_month, months)
        ]
        return torch.stack(rows)

    def forward(self, time_features: torch.Tensor) -> torch.Tensor:
        """Project raw time features to d_model.

        Args:
            time_features: [B, 9] raw cyclic features

        Returns:
            [B, d_model] conditioning bias
        """
        return self.proj(time_features)
