"""Full Bash-MANTIS-0.6 model integrating all subsystems.

Combines:
- Ternary transformer backbone
- Manifold state
- Structured workspace
- Doc-to-LoRA memory
- Preference head
"""

from __future__ import annotations

import torch
import torch.nn as nn
from dataclasses import dataclass

from bash_mantis.models.transformer import TransformerBackbone
from bash_mantis.models.manifold import ManifoldState
from bash_mantis.models.workspace import Workspace
from bash_mantis.models.lora_memory import DocToLoRA
from bash_mantis.models.preference_head import PreferenceHead


@dataclass
class MantisOutput:
    """Output container for BashMantisModel."""
    logits: torch.Tensor                    # [B, T, vocab_size]
    hidden: torch.Tensor                    # [B, T, d_model]
    kappa: torch.Tensor                     # [B, manifold_dim]
    slots: torch.Tensor                     # [B, n_slots, slot_dim]
    lora_A: torch.Tensor | None = None      # [B, d_model, rank]
    lora_B: torch.Tensor | None = None      # [B, rank, d_model]


class BashMantisModel(nn.Module):
    """Bash-MANTIS-0.6: Tiny Ternary Instruction-Following Shell Model.

    This model processes structured shell task prompts and generates
    Bash commands through five integrated subsystems.
    """

    def __init__(
        self,
        vocab_size: int = 320,
        d_model: int = 108,
        n_heads: int = 4,
        n_layers: int = 4,
        ffn_dim: int = 256,
        ctx_len: int = 256,
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
    ):
        super().__init__()
        self.d_model = d_model
        self.lora_target_block = lora_target_block

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

        # 3. Workspace
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

    def forward(
        self,
        input_ids: torch.Tensor,
        doc_ids: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        return_manifold_intermediates: bool = False,
    ) -> MantisOutput:
        """Full forward pass through all subsystems.

        Args:
            input_ids: [B, T] token IDs
            doc_ids: [B, doc_len] optional rule card token IDs
            mask: [B, 1, T, T] optional attention mask
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
            # For batch: we need to handle per-sample LoRA
            # Use mean LoRA for simplicity in training
            self.doc_memory.inject_into_block(
                target_block, lora_A.mean(0), lora_B.mean(0)
            )

        # Run backbone
        logits, hidden = self.backbone(input_ids, mask)

        # Clear LoRA after forward
        if doc_ids is not None:
            self.doc_memory.clear_from_block(
                self.backbone.blocks[self.lora_target_block]
            )

        # Initialize manifold and workspace
        kappa = self.manifold.init_state(B, device)
        slots = self.workspace.init_slots(B, device)

        # Run manifold + workspace updates using hidden states
        # Process at a few key positions for efficiency
        n_steps = min(4, T)
        step_positions = torch.linspace(0, T - 1, n_steps).long()

        kappa_trajectory = [kappa] if return_manifold_intermediates else None

        for pos in step_positions:
            h_t = hidden[:, pos, :]  # [B, d_model]

            # Update workspace
            slots = self.workspace.update(slots, h_t, kappa)
            p_bar = self.workspace.pool(slots)  # [B, slot_dim]

            # Memory context
            m_t = None
            if doc_ids is not None:
                doc_vec = self.doc_memory.encoder(doc_ids)
                # Project doc_vec to d_model
                m_t = torch.zeros(B, self.d_model, device=device)
                m_t[:, : doc_vec.shape[-1]] = doc_vec

            # Update manifold
            kappa = self.manifold.step(kappa, h_t, p_bar, m_t)

            if return_manifold_intermediates:
                kappa_trajectory.append(kappa)

        output = MantisOutput(
            logits=logits,
            hidden=hidden,
            kappa=kappa,
            slots=slots,
            lora_A=lora_A,
            lora_B=lora_B,
        )

        if return_manifold_intermediates:
            output._kappa_trajectory = kappa_trajectory

        return output

    def get_sequence_representation(self, hidden: torch.Tensor) -> torch.Tensor:
        """Get a single representation vector for a sequence.

        Uses mean pooling over non-padding positions.
        """
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
        )
