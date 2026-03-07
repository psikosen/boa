"""Manifold state module for Bash-MANTIS.

Implements a small recurrent latent vector kappa that tracks reasoning state
while the model processes instructions and generates shell commands.

The 12-dimensional state tracks:
  0: task mode
  1: command family
  2: path/file operation state
  3: pipeline composition state
  4: quoting/escaping stability
  5: loop/conditional mode
  6: function-scope mode
  7: repair confidence
  8: redirection state
  9: argument-binding state
  10: safety mode
  11: completion confidence
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ManifoldState(nn.Module):
    """Recurrent manifold state with gated update.

    Update rule:
        kappa_{t+1} = kappa_t + gate * MLP([kappa_t; h_t; p_bar_t; m_t])

    where:
        h_t = current hidden state from transformer
        p_bar_t = pooled workspace state
        m_t = memory context (from doc-to-LoRA)
        gate = learned gating to control update magnitude
    """

    def __init__(self, manifold_dim: int = 12, hidden_dim: int = 108, workspace_dim: int = 24):
        super().__init__()
        self.manifold_dim = manifold_dim

        # Input projection: concat of [kappa, h_proj, p_bar, m_proj]
        input_dim = manifold_dim + manifold_dim + workspace_dim + manifold_dim

        self.h_proj = nn.Linear(hidden_dim, manifold_dim)
        self.m_proj = nn.Linear(hidden_dim, manifold_dim)

        # Update MLP
        self.update_mlp = nn.Sequential(
            nn.Linear(input_dim, manifold_dim * 4),
            nn.GELU(),
            nn.Linear(manifold_dim * 4, manifold_dim),
        )

        # Gating
        self.gate_mlp = nn.Sequential(
            nn.Linear(input_dim, manifold_dim),
            nn.Sigmoid(),
        )

        # Initial state
        self.kappa_init = nn.Parameter(torch.zeros(manifold_dim))

    def init_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Initialize kappa for a batch."""
        return self.kappa_init.unsqueeze(0).expand(batch_size, -1).clone()

    def step(
        self,
        kappa: torch.Tensor,
        h_t: torch.Tensor,
        p_bar: torch.Tensor,
        m_t: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Perform one manifold update step.

        Args:
            kappa: current state [B, manifold_dim]
            h_t: transformer hidden [B, hidden_dim]
            p_bar: pooled workspace [B, workspace_dim]
            m_t: memory context [B, hidden_dim] or None
        """
        h_proj = self.h_proj(h_t)

        if m_t is not None:
            m_proj = self.m_proj(m_t)
        else:
            m_proj = torch.zeros_like(h_proj)

        combined = torch.cat([kappa, h_proj, p_bar, m_proj], dim=-1)

        update = self.update_mlp(combined)
        gate = self.gate_mlp(combined)

        kappa_new = kappa + gate * update
        return kappa_new

    def compute_jacobian_norm(self, kappa: torch.Tensor, h_t: torch.Tensor,
                               p_bar: torch.Tensor, m_t: torch.Tensor | None = None,
                               n_power_iters: int = 3) -> torch.Tensor:
        """Approximate the max singular value of the Jacobian dF/dkappa.

        Uses power iteration for efficiency. This is used in the contraction loss.
        """
        batch_size = kappa.shape[0]
        # Random vector for power iteration
        v = torch.randn_like(kappa)
        v = v / (v.norm(dim=-1, keepdim=True) + 1e-8)

        for _ in range(n_power_iters):
            kappa_var = kappa.detach().requires_grad_(True)
            kappa_next = self.step(kappa_var, h_t, p_bar, m_t)
            # Compute Jv via vjp
            Jv = torch.autograd.grad(
                kappa_next, kappa_var,
                grad_outputs=v,
                create_graph=True,
                retain_graph=True,
            )[0]
            sigma = Jv.norm(dim=-1, keepdim=True)
            v = Jv / (sigma + 1e-8)

        return sigma.squeeze(-1).mean()
