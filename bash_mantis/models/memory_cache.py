"""Multi-turn Memory Caching for Bash-MANTIS.

Caches manifold (kappa) and workspace slot states at turn boundaries,
then uses GRM-style gated aggregation to let new turns selectively
recall from all prior cached checkpoints.

Each "turn" corresponds to one user→model exchange (e.g. a repair loop
iteration).  At the end of each turn the final kappa and slots are
appended to the cache.  At the start of the next turn, the model queries
the cache to recover relevant long-range context that would otherwise be
lost when kappa is re-initialised.

Design choices following the paper:
  - Separate u projection for gate selection (not shared with retrieval q)
  - Content-aware scalar gates via dot-product similarity
  - Softmax-normalised gates across cached turns
  - Cache grows linearly with number of turns (cheap for interactive use)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MemoryCache(nn.Module):
    """Turn-level memory cache with GRM-gated retrieval.

    Stores (kappa, pooled_slots) pairs at turn boundaries and produces
    a bias vector that is added to the current turn's kappa after init.
    """

    def __init__(
        self,
        manifold_dim: int = 12,
        slot_dim: int = 24,
        max_cached_turns: int = 32,
    ):
        super().__init__()
        self.manifold_dim = manifold_dim
        self.slot_dim = slot_dim
        self.checkpoint_dim = manifold_dim + slot_dim  # what we cache per turn
        self.max_cached_turns = max_cached_turns

        # Gate projection: maps current hidden → checkpoint_dim for
        # dot-product similarity with cached checkpoints.
        # This is the *u* projection — deliberately separate from any
        # retrieval query (the paper's critical ablation finding).
        self.gate_u_proj = nn.Linear(manifold_dim + slot_dim, self.checkpoint_dim)

        # Output projection: maps aggregated checkpoint back to
        # separate kappa-bias and slots-bias.
        self.out_kappa_proj = nn.Linear(self.checkpoint_dim, manifold_dim)
        self.out_slots_proj = nn.Linear(self.checkpoint_dim, slot_dim)

    # ------------------------------------------------------------------
    # Cache management (not nn.Parameters — pure inference-time state)
    # ------------------------------------------------------------------

    @staticmethod
    def new_cache(device: torch.device) -> dict:
        """Create an empty cache dict for one sample.

        Returns a plain dict so callers can manage it outside the module.
        """
        return {"checkpoints": [], "device": device}

    @staticmethod
    def append(
        cache: dict,
        kappa: torch.Tensor,
        pooled_slots: torch.Tensor,
    ) -> None:
        """Append a (kappa, pooled_slots) checkpoint to the cache.

        Args:
            cache: cache dict from ``new_cache``
            kappa: [manifold_dim] final kappa of the concluded turn
            pooled_slots: [slot_dim] mean-pooled workspace slots
        """
        ckpt = torch.cat([kappa.detach(), pooled_slots.detach()], dim=-1)
        cache["checkpoints"].append(ckpt)
        # Enforce max size — drop oldest if exceeded
        if len(cache["checkpoints"]) > 32:
            cache["checkpoints"] = cache["checkpoints"][-32:]

    # ------------------------------------------------------------------
    # GRM-gated retrieval
    # ------------------------------------------------------------------

    def retrieve(
        self,
        cache: dict,
        kappa: torch.Tensor,
        pooled_slots: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Query the cache and return bias vectors for kappa and slots.

        If the cache is empty, returns zeros.

        Args:
            cache: cache dict with prior checkpoints
            kappa: [manifold_dim] current turn's initial kappa
            pooled_slots: [slot_dim] current turn's initial pooled slots

        Returns:
            kappa_bias:  [manifold_dim] additive bias for kappa
            slots_bias:  [slot_dim] additive bias for pooled slot readout
        """
        if not cache["checkpoints"]:
            return (
                torch.zeros(self.manifold_dim, device=kappa.device),
                torch.zeros(self.slot_dim, device=kappa.device),
            )

        # Stack cached checkpoints: [N, checkpoint_dim]
        cached = torch.stack(cache["checkpoints"])

        # Current context vector
        ctx = torch.cat([kappa, pooled_slots], dim=-1)  # [checkpoint_dim]

        # u is the gate query — separate projection
        u = self.gate_u_proj(ctx)  # [checkpoint_dim]

        # Dot-product similarity with each cached checkpoint
        gamma = cached @ u  # [N]
        gamma = F.softmax(gamma, dim=0)  # normalise across turns

        # Weighted sum of cached checkpoints
        aggregated = (gamma.unsqueeze(-1) * cached).sum(dim=0)  # [checkpoint_dim]

        kappa_bias = self.out_kappa_proj(aggregated)
        slots_bias = self.out_slots_proj(aggregated)
        return kappa_bias, slots_bias

    def retrieve_batch(
        self,
        caches: list[dict],
        kappa: torch.Tensor,
        pooled_slots: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Batched retrieval across multiple samples.

        Args:
            caches: list of B cache dicts
            kappa: [B, manifold_dim]
            pooled_slots: [B, slot_dim]

        Returns:
            kappa_bias:  [B, manifold_dim]
            slots_bias:  [B, slot_dim]
        """
        kappa_biases = []
        slots_biases = []
        for i, cache in enumerate(caches):
            kb, sb = self.retrieve(cache, kappa[i], pooled_slots[i])
            kappa_biases.append(kb)
            slots_biases.append(sb)
        return torch.stack(kappa_biases), torch.stack(slots_biases)
