"""Ternary preference head for Bash-MANTIS.

Implements win / tie / lose ranking for candidate Bash outputs.
The tie class is essential because many shell commands are different
strings with equivalent observable behavior.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PreferenceHead(nn.Module):
    """Three-way preference classifier: win, tie, lose.

    Given two candidate representations, predicts whether:
    - candidate A is better (win)
    - candidates are equivalent (tie)
    - candidate B is better (lose)
    """

    def __init__(self, d_model: int = 108, hidden_dim: int = 48):
        super().__init__()
        # Takes concatenation of two candidate representations
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),  # win, tie, lose
        )

    def forward(
        self,
        rep_a: torch.Tensor,
        rep_b: torch.Tensor,
    ) -> torch.Tensor:
        """Classify preference between two candidates.

        Args:
            rep_a: [B, d_model] representation of candidate A
            rep_b: [B, d_model] representation of candidate B

        Returns:
            logits: [B, 3] for (win, tie, lose)
        """
        combined = torch.cat([rep_a, rep_b], dim=-1)
        return self.classifier(combined)

    def compute_loss(
        self,
        rep_a: torch.Tensor,
        rep_b: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute cross-entropy loss for preference prediction.

        Args:
            rep_a: [B, d_model]
            rep_b: [B, d_model]
            labels: [B] with values in {0=win, 1=tie, 2=lose}
        """
        logits = self.forward(rep_a, rep_b)
        return F.cross_entropy(logits, labels)

    def rank_candidates(
        self,
        candidates: list[torch.Tensor],
    ) -> list[tuple[int, float]]:
        """Rank multiple candidates using pairwise comparisons.

        Args:
            candidates: list of [d_model] tensors

        Returns:
            List of (index, score) sorted by score descending.
            Score is the win rate across all pairwise comparisons.
        """
        n = len(candidates)
        scores = [0.0] * n

        for i in range(n):
            for j in range(i + 1, n):
                logits = self.forward(
                    candidates[i].unsqueeze(0),
                    candidates[j].unsqueeze(0),
                )
                probs = F.softmax(logits, dim=-1).squeeze(0)
                # win prob for i, tie splits evenly, lose prob for j
                scores[i] += probs[0].item() + 0.5 * probs[1].item()
                scores[j] += probs[2].item() + 0.5 * probs[1].item()

        ranked = sorted(enumerate(scores), key=lambda x: -x[1])
        return ranked
