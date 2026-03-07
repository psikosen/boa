"""Latent space analysis for Bash-MANTIS manifold state.

Analyzes whether:
- Correct terminal states are more stable than incorrect ones
- Equivalent commands cluster more closely than inequivalent ones
- Text prompts and correct Bash solutions align in latent space
"""

from __future__ import annotations

import torch
import numpy as np
from dataclasses import dataclass


@dataclass
class LatentMetrics:
    """Metrics about latent space structure."""
    mean_attractor_residual_correct: float = 0.0
    mean_attractor_residual_incorrect: float = 0.0
    mean_equiv_distance: float = 0.0
    mean_non_equiv_distance: float = 0.0
    mean_text_bash_distance: float = 0.0
    stability_ratio: float = 0.0  # correct stability / incorrect stability

    def to_dict(self) -> dict:
        return {
            "mean_attractor_residual_correct": self.mean_attractor_residual_correct,
            "mean_attractor_residual_incorrect": self.mean_attractor_residual_incorrect,
            "mean_equiv_distance": self.mean_equiv_distance,
            "mean_non_equiv_distance": self.mean_non_equiv_distance,
            "mean_text_bash_distance": self.mean_text_bash_distance,
            "stability_ratio": self.stability_ratio,
        }


class LatentAnalyzer:
    """Analyze manifold state geometry for Bash-MANTIS."""

    @staticmethod
    def pairwise_distances(kappas: torch.Tensor) -> torch.Tensor:
        """Compute pairwise L2 distances between kappa vectors.

        Args:
            kappas: [N, manifold_dim]

        Returns:
            [N, N] distance matrix
        """
        diff = kappas.unsqueeze(0) - kappas.unsqueeze(1)
        return diff.pow(2).sum(-1).sqrt()

    @staticmethod
    def attractor_residual(
        kappa: torch.Tensor,
        manifold: torch.nn.Module,
        h_t: torch.Tensor,
        p_bar: torch.Tensor,
        m_t: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute ||F(kappa) - kappa||^2 for each sample.

        Small values mean kappa is near a fixed point (attractor).
        """
        with torch.no_grad():
            kappa_next = manifold.step(kappa, h_t, p_bar, m_t)
        return (kappa_next - kappa).pow(2).sum(-1)

    def analyze(
        self,
        kappa_correct: torch.Tensor,
        kappa_incorrect: torch.Tensor | None = None,
        kappa_equiv_pairs: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        kappa_non_equiv_pairs: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        kappa_text: torch.Tensor | None = None,
        kappa_bash: torch.Tensor | None = None,
        manifold: torch.nn.Module | None = None,
        h_t: torch.Tensor | None = None,
        p_bar: torch.Tensor | None = None,
    ) -> LatentMetrics:
        """Run full latent analysis."""
        metrics = LatentMetrics()

        # Attractor stability for correct states
        if manifold is not None and h_t is not None and p_bar is not None:
            residuals_correct = self.attractor_residual(
                kappa_correct, manifold, h_t, p_bar
            )
            metrics.mean_attractor_residual_correct = residuals_correct.mean().item()

            if kappa_incorrect is not None:
                residuals_incorrect = self.attractor_residual(
                    kappa_incorrect, manifold, h_t[:kappa_incorrect.shape[0]],
                    p_bar[:kappa_incorrect.shape[0]],
                )
                metrics.mean_attractor_residual_incorrect = residuals_incorrect.mean().item()
                if metrics.mean_attractor_residual_incorrect > 0:
                    metrics.stability_ratio = (
                        metrics.mean_attractor_residual_correct
                        / metrics.mean_attractor_residual_incorrect
                    )

        # Equivalence clustering
        if kappa_equiv_pairs:
            equiv_dists = [
                (ki - kj).pow(2).sum().sqrt().item()
                for ki, kj in kappa_equiv_pairs
            ]
            metrics.mean_equiv_distance = np.mean(equiv_dists)

        if kappa_non_equiv_pairs:
            non_equiv_dists = [
                (ki - kj).pow(2).sum().sqrt().item()
                for ki, kj in kappa_non_equiv_pairs
            ]
            metrics.mean_non_equiv_distance = np.mean(non_equiv_dists)

        # Text-Bash alignment
        if kappa_text is not None and kappa_bash is not None:
            text_bash_dist = (kappa_text - kappa_bash).pow(2).sum(-1).sqrt()
            metrics.mean_text_bash_distance = text_bash_dist.mean().item()

        return metrics
