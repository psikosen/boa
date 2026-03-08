"""Structured workspace for Bash-MANTIS.

Implements 6 persistent slots that store task entities:
  0: action / command intent
  1: targets / files / paths
  2: filters / predicates
  3: pipes / redirection
  4: variables / control flow
  5: function summary / repair state

Each slot is a small vector updated via gated attention from the
transformer hidden states and manifold state.

Slot reads use GRM-style content-aware gating: each slot's contribution
is weighted by a similarity score between a dedicated gate projection
(u_t, separate from the retrieval query) and the slot content.  This
lets the model selectively suppress irrelevant slots per-token.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Workspace(nn.Module):
    """Persistent slot-based workspace with gated updates and GRM reads."""

    def __init__(
        self,
        n_slots: int = 6,
        slot_dim: int = 24,
        hidden_dim: int = 108,
        manifold_dim: int = 12,
    ):
        super().__init__()
        self.n_slots = n_slots
        self.slot_dim = slot_dim

        # Initial slot values (learned)
        self.slot_init = nn.Parameter(torch.randn(n_slots, slot_dim) * 0.02)

        # Attention: query from slots, key/value from hidden + manifold
        self.slot_to_query = nn.Linear(slot_dim, hidden_dim)
        self.hidden_to_key = nn.Linear(hidden_dim, hidden_dim)
        self.hidden_to_value = nn.Linear(hidden_dim, slot_dim)

        # Manifold-conditioned gating
        self.gate_proj = nn.Linear(manifold_dim + slot_dim, slot_dim)

        # Slot update MLP
        self.update_mlp = nn.Sequential(
            nn.Linear(slot_dim * 2, slot_dim * 2),
            nn.GELU(),
            nn.Linear(slot_dim * 2, slot_dim),
        )

        # --- GRM read gating ---
        # Separate u projection for gate selection (MUST differ from the
        # retrieval query per the paper's ablation: shared u/q → 0.0%).
        self.read_u_proj = nn.Linear(hidden_dim, slot_dim)

    def init_slots(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Initialize workspace slots for a batch. Returns [B, n_slots, slot_dim]."""
        return self.slot_init.unsqueeze(0).expand(batch_size, -1, -1).clone()

    def update(
        self,
        slots: torch.Tensor,
        hidden: torch.Tensor,
        kappa: torch.Tensor,
    ) -> torch.Tensor:
        """Update workspace slots given current hidden states and manifold state.

        Args:
            slots: [B, n_slots, slot_dim]
            hidden: [B, seq_len, hidden_dim] or [B, hidden_dim]
            kappa: [B, manifold_dim]

        Returns:
            Updated slots [B, n_slots, slot_dim]
        """
        if hidden.dim() == 2:
            hidden = hidden.unsqueeze(1)  # [B, 1, hidden_dim]

        # Cross-attention: slots attend to hidden states
        Q = self.slot_to_query(slots)  # [B, n_slots, hidden_dim]
        K = self.hidden_to_key(hidden)  # [B, seq_len, hidden_dim]
        V = self.hidden_to_value(hidden)  # [B, seq_len, slot_dim]

        scale = Q.shape[-1] ** 0.5
        attn = torch.bmm(Q, K.transpose(1, 2)) / scale  # [B, n_slots, seq_len]
        attn = F.softmax(attn, dim=-1)
        attended = torch.bmm(attn, V)  # [B, n_slots, slot_dim]

        # Gated update conditioned on manifold state
        kappa_expanded = kappa.unsqueeze(1).expand(-1, self.n_slots, -1)
        gate_input = torch.cat([kappa_expanded, slots], dim=-1)
        gate = torch.sigmoid(self.gate_proj(gate_input))

        update_input = torch.cat([slots, attended], dim=-1)
        update = self.update_mlp(update_input)

        new_slots = slots + gate * update
        return new_slots

    def read(self, slots: torch.Tensor, h_t: torch.Tensor) -> torch.Tensor:
        """GRM-gated read from workspace slots.

        Each slot's contribution is weighted by the dot-product similarity
        between a dedicated gate vector u_t (projected from h_t via a
        *separate* projection from the retrieval query) and the slot
        content.  This produces a per-token, content-aware pooled vector.

        Args:
            slots: [B, n_slots, slot_dim]
            h_t: [B, hidden_dim] — current hidden state

        Returns:
            [B, slot_dim] — GRM-gated slot readout
        """
        # u_t is the *gate* query — separate from any retrieval query
        u_t = self.read_u_proj(h_t)  # [B, slot_dim]

        # Similarity between u_t and each slot → per-slot scalar gate
        # slots: [B, n_slots, slot_dim], u_t: [B, 1, slot_dim]
        gamma = torch.bmm(
            slots, u_t.unsqueeze(-1)
        ).squeeze(-1)  # [B, n_slots]
        gamma = F.softmax(gamma, dim=-1)  # normalise across slots

        # Weighted combination of slot contents
        # gamma: [B, n_slots, 1] * slots: [B, n_slots, slot_dim] → sum
        readout = (gamma.unsqueeze(-1) * slots).sum(dim=1)  # [B, slot_dim]
        return readout

    def pool(self, slots: torch.Tensor) -> torch.Tensor:
        """Pool workspace slots to a single vector (mean). Returns [B, slot_dim]."""
        return slots.mean(dim=1)
