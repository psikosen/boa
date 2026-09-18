"""Full Bash-MANTIS-0.6 model integrating all subsystems.

Combines:
- Ternary transformer backbone
- Manifold state
- Structured workspace (with GRM-gated reads)
- Doc-to-LoRA memory
- Preference head
- Temporal context conditioning
- Multi-turn memory cache (manifold + workspace checkpoints)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from dataclasses import dataclass, field

from bash_mantis.models.transformer import TransformerBackbone
from bash_mantis.models.manifold import ManifoldState
from bash_mantis.models.workspace import Workspace
from bash_mantis.models.lora_memory import DocToLoRA
from bash_mantis.models.preference_head import PreferenceHead
from bash_mantis.models.time_context import TimeContext
from bash_mantis.models.memory_cache import MemoryCache
from bash_mantis.models.typed_heads import TypedManifoldHeads
from bash_mantis.models.tool_gate import ToolCallGate


@dataclass
class MantisOutput:
    """Output container for BashMantisModel."""
    logits: torch.Tensor                    # [B, T, vocab_size]
    hidden: torch.Tensor                    # [B, T, d_model]
    kappa: torch.Tensor                     # [B, manifold_dim]
    slots: torch.Tensor                     # [B, n_slots, slot_dim]
    slot_readout: torch.Tensor | None = None  # [B, slot_dim] GRM-gated read
    lora_A: torch.Tensor | None = None      # [B, d_model, rank]
    lora_B: torch.Tensor | None = None      # [B, rank, d_model]
    #: Calibrated typed answers read off kappa's declared dimensions:
    #: repair_confidence, safety_risk, completion_score, completion_confidence.
    typed: dict[str, torch.Tensor] | None = None


class BashMantisModel(nn.Module):
    """Bash-MANTIS-0.6: Tiny Ternary Instruction-Following Shell Model.

    This model processes structured shell task prompts and generates
    Bash commands through seven integrated subsystems.
    """

    def __init__(
        self,
        vocab_size: int = 320,
        d_model: int = 108,
        n_heads: int = 4,
        n_layers: int = 4,
        ffn_dim: int = 256,
        ctx_len: int = 512,
        dropout: float = 0.1,
        manifold_dim: int = 12,
        workspace_slots: int = 6,
        workspace_slot_dim: int = 24,
        lora_rank: int = 2,
        doc_encoder_dim: int = 64,
        doc_max_len: int = 128,
        lora_target_block: int = 2,
        pref_hidden_dim: int = 48,
        ternary_enabled: bool = False,
        precision_island_size: int = 16,
        time_conditioning: bool = True,
        mc_max_cached_turns: int = 32,
        gate_confidence_threshold: float = 0.8,
    ):
        super().__init__()
        self.d_model = d_model
        self.lora_target_block = lora_target_block
        self.time_conditioning = time_conditioning

        # 1. Ternary backbone
        self.backbone = TransformerBackbone(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            ffn_dim=ffn_dim,
            ctx_len=ctx_len,
            ternary=ternary_enabled,
            precision_island_size=precision_island_size,
            dropout=dropout,
        )

        # 2. Manifold state
        self.manifold = ManifoldState(
            manifold_dim=manifold_dim,
            hidden_dim=d_model,
            workspace_dim=workspace_slot_dim,
        )

        # 3. Workspace (with GRM-gated reads)
        self.workspace = Workspace(
            n_slots=workspace_slots,
            slot_dim=workspace_slot_dim,
            hidden_dim=d_model,
            manifold_dim=manifold_dim,
        )

        # 4. Doc-to-LoRA memory
        self.doc_memory = DocToLoRA(
            vocab_size=vocab_size,
            doc_dim=doc_encoder_dim,
            doc_max_len=doc_max_len,
            d_model=d_model,
            rank=lora_rank,
        )

        # 5. Preference head
        self.preference = PreferenceHead(
            d_model=d_model,
            hidden_dim=pref_hidden_dim,
        )

        # 6. Temporal context conditioning
        self.time_ctx = TimeContext(d_model=d_model) if time_conditioning else None

        # 7. Multi-turn memory cache
        self.memory_cache = MemoryCache(
            manifold_dim=manifold_dim,
            slot_dim=workspace_slot_dim,
            max_cached_turns=mc_max_cached_turns,
        )

        # 8. Calibrated typed heads over kappa's declared dimensions
        self.typed_heads = TypedManifoldHeads()

        # 9. Tool-call gate (routing decision over the manifold state)
        self.tool_gate = ToolCallGate(
            manifold_dim=manifold_dim,
            confidence_threshold=gate_confidence_threshold,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        doc_ids: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        time_features: torch.Tensor | None = None,
        turn_caches: list[dict] | None = None,
        return_manifold_intermediates: bool = False,
    ) -> MantisOutput:
        """Full forward pass through all subsystems.

        Args:
            input_ids: [B, T] token IDs
            doc_ids: [B, doc_len] optional rule card token IDs
            mask: [B, 1, T, T] optional attention mask
            time_features: [B, 9] optional raw temporal features from
                TimeContext.encode_time_batch.  When provided and
                time_conditioning is enabled, the projected time vector
                is added to every token embedding before the backbone.
            turn_caches: optional list of B cache dicts (one per sample)
                from MemoryCache.new_cache().  When provided, the cache
                is queried at the start of the turn and the final
                kappa/slots are appended after processing.
            return_manifold_intermediates: if True, track kappa trajectory

        Returns:
            MantisOutput with all outputs
        """
        B, T = input_ids.shape
        device = input_ids.device

        # Optional: inject LoRA from rule card
        lora_A = lora_B = None
        if doc_ids is not None:
            lora_A, lora_B = self.doc_memory(doc_ids)
            target_block = self.backbone.blocks[self.lora_target_block]
            self.doc_memory.inject_into_block(
                target_block, lora_A.mean(0), lora_B.mean(0)
            )

        # --- Time conditioning: add temporal bias to embeddings ---
        # We reach into the backbone to add the bias *before* the
        # transformer blocks run, so every layer sees the time signal.
        if self.time_ctx is not None and time_features is not None:
            time_bias = self.time_ctx(time_features)  # [B, d_model]
            # Temporarily patch the backbone embedding to include time
            raw_emb = self.backbone.tok_emb(input_ids)  # [B, T, d_model]
            raw_emb = raw_emb + time_bias.unsqueeze(1)  # broadcast across T
            logits, hidden = self._backbone_from_embeddings(raw_emb, mask)
        else:
            logits, hidden = self.backbone(input_ids, mask)

        # Clear LoRA after forward
        if doc_ids is not None:
            self.doc_memory.clear_from_block(
                self.backbone.blocks[self.lora_target_block]
            )

        # Initialize manifold and workspace
        kappa = self.manifold.init_state(B, device)
        slots = self.workspace.init_slots(B, device)

        # --- Multi-turn MC: query cached checkpoints to bias init ---
        if turn_caches is not None:
            p_bar_init = self.workspace.pool(slots)
            kappa_bias, slots_bias = self.memory_cache.retrieve_batch(
                turn_caches, kappa, p_bar_init,
            )
            kappa = kappa + kappa_bias
            # Apply slots_bias to pooled representation for manifold,
            # but we also distribute it back across individual slots
            # (uniform additive — the GRM read will re-weight later).
            slots = slots + slots_bias.unsqueeze(1).expand_as(slots) / slots.shape[1]

        # Run manifold + workspace updates using hidden states
        n_steps = min(4, T)
        step_positions = torch.linspace(0, T - 1, n_steps).long()

        kappa_trajectory = [kappa] if return_manifold_intermediates else None

        for pos in step_positions:
            h_t = hidden[:, pos, :]  # [B, d_model]

            # Update workspace slots
            slots = self.workspace.update(slots, h_t, kappa)

            # GRM-gated read (content-aware slot weighting)
            p_bar = self.workspace.read(slots, h_t)  # [B, slot_dim]

            # Memory context
            m_t = None
            if doc_ids is not None:
                doc_vec = self.doc_memory.encoder(doc_ids)
                m_t = torch.zeros(B, self.d_model, device=device)
                m_t[:, : doc_vec.shape[-1]] = doc_vec

            # Update manifold
            kappa = self.manifold.step(kappa, h_t, p_bar, m_t)

            if return_manifold_intermediates:
                kappa_trajectory.append(kappa)

        # --- Multi-turn MC: cache final state for next turn ---
        if turn_caches is not None:
            p_bar_final = self.workspace.pool(slots)
            for i, cache in enumerate(turn_caches):
                MemoryCache.append(cache, kappa[i], p_bar_final[i])

        # Final GRM-gated readout using last hidden position
        h_last = hidden[:, -1, :]
        slot_readout = self.workspace.read(slots, h_last)

        output = MantisOutput(
            logits=logits,
            hidden=hidden,
            kappa=kappa,
            slots=slots,
            slot_readout=slot_readout,
            lora_A=lora_A,
            lora_B=lora_B,
            typed=self.typed_heads(kappa),
        )

        if return_manifold_intermediates:
            output._kappa_trajectory = kappa_trajectory

        return output

    def _backbone_from_embeddings(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the backbone transformer from pre-computed embeddings.

        This bypasses the token embedding layer so we can inject the
        time-conditioning bias before the first transformer block.
        """
        B, T, _ = x.shape
        if mask is None:
            mask = torch.tril(torch.ones(T, T, device=x.device)).unsqueeze(0).unsqueeze(0)

        cos, sin = self.backbone.rotary(T)
        for block in self.backbone.blocks:
            x = block(x, cos, sin, mask)

        hidden = self.backbone.final_norm(x)
        logits = self.backbone.lm_head(hidden)
        return logits, hidden

    def get_sequence_representation(self, hidden: torch.Tensor) -> torch.Tensor:
        """Get a single representation vector for a sequence."""
        return hidden.mean(dim=1)

    def enable_ternary(self):
        self.backbone.enable_ternary()

    def disable_ternary(self):
        self.backbone.disable_ternary()

    @classmethod
    def from_config(cls, config) -> "BashMantisModel":
        """Create model from a MantisConfig."""
        mc = config.model
        return cls(
            vocab_size=mc.vocab_size,
            d_model=mc.d_model,
            n_heads=mc.n_heads,
            n_layers=mc.n_layers,
            ffn_dim=mc.ffn_dim,
            ctx_len=mc.ctx_len,
            dropout=mc.dropout,
            manifold_dim=mc.manifold_dim,
            workspace_slots=mc.workspace_slots,
            workspace_slot_dim=mc.workspace_slot_dim,
            lora_rank=mc.lora_rank,
            doc_encoder_dim=mc.doc_encoder_dim,
            doc_max_len=mc.doc_max_len,
            lora_target_block=mc.lora_target_block,
            pref_hidden_dim=mc.pref_hidden_dim,
            ternary_enabled=mc.ternary_enabled,
            precision_island_size=mc.precision_island_size,
            time_conditioning=mc.time_conditioning,
            mc_max_cached_turns=mc.mc_max_cached_turns,
            gate_confidence_threshold=getattr(
                mc, "gate_confidence_threshold", 0.8),
        )
