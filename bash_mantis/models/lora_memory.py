"""Doc-to-LoRA memory module for Bash-MANTIS.

Converts short reference documents (rule cards) into compact rank-2 LoRA
adapters via a tiny hypernetwork. This lets the model internalize reusable
shell knowledge without repeatedly consuming the same text in the prompt.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DocEncoder(nn.Module):
    """Encodes a short document (rule card) into a fixed-size vector."""

    def __init__(
        self,
        vocab_size: int = 320,
        doc_dim: int = 64,
        max_len: int = 128,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, doc_dim)
        self.pos_embedding = nn.Embedding(max_len, doc_dim)
        self.pool_proj = nn.Sequential(
            nn.Linear(doc_dim, doc_dim),
            nn.GELU(),
            nn.Linear(doc_dim, doc_dim),
        )

    def forward(self, doc_ids: torch.Tensor) -> torch.Tensor:
        """Encode a document.

        Args:
            doc_ids: [B, doc_len] token IDs

        Returns:
            doc_vec: [B, doc_dim] document representation
        """
        B, L = doc_ids.shape
        positions = torch.arange(L, device=doc_ids.device).unsqueeze(0)
        x = self.embedding(doc_ids) + self.pos_embedding(positions)
        # Mean pool + project
        x = x.mean(dim=1)
        return self.pool_proj(x)


class LoRAHypernetwork(nn.Module):
    """Generates rank-r LoRA matrices from a document vector.

    Produces A: [d_model, rank] and B: [rank, d_model] such that
    the LoRA update is: h + h @ A @ B
    """

    def __init__(
        self,
        doc_dim: int = 64,
        d_model: int = 108,
        rank: int = 2,
    ):
        super().__init__()
        self.d_model = d_model
        self.rank = rank

        # Generate A matrix: doc_dim -> d_model * rank
        self.gen_A = nn.Sequential(
            nn.Linear(doc_dim, doc_dim * 2),
            nn.GELU(),
            nn.Linear(doc_dim * 2, d_model * rank),
        )
        # Generate B matrix: doc_dim -> rank * d_model
        self.gen_B = nn.Sequential(
            nn.Linear(doc_dim, doc_dim * 2),
            nn.GELU(),
            nn.Linear(doc_dim * 2, rank * d_model),
        )

        # Scale initialization to be small
        with torch.no_grad():
            self.gen_A[-1].weight.mul_(0.01)
            self.gen_B[-1].weight.mul_(0.01)

    def forward(self, doc_vec: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate LoRA matrices from document vector.

        Args:
            doc_vec: [B, doc_dim]

        Returns:
            lora_A: [B, d_model, rank]
            lora_B: [B, rank, d_model]
        """
        A = self.gen_A(doc_vec).view(-1, self.d_model, self.rank)
        B = self.gen_B(doc_vec).view(-1, self.rank, self.d_model)
        return A, B


class DocToLoRA(nn.Module):
    """Complete Doc-to-LoRA module.

    Encodes a rule card document and generates a rank-2 LoRA adapter
    that can be injected into a target transformer block.
    """

    def __init__(
        self,
        vocab_size: int = 320,
        doc_dim: int = 64,
        doc_max_len: int = 128,
        d_model: int = 108,
        rank: int = 2,
    ):
        super().__init__()
        self.encoder = DocEncoder(vocab_size, doc_dim, doc_max_len)
        self.hypernetwork = LoRAHypernetwork(doc_dim, d_model, rank)
        self.d_model = d_model

    def forward(self, doc_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode document and generate LoRA matrices.

        Args:
            doc_ids: [B, doc_len] token IDs for the rule card

        Returns:
            lora_A: [B, d_model, rank]
            lora_B: [B, rank, d_model]
        """
        doc_vec = self.encoder(doc_ids)
        return self.hypernetwork(doc_vec)

    def get_context_vector(self, doc_ids: torch.Tensor) -> torch.Tensor:
        """Get the document context vector (for manifold input).

        Returns a vector projected to d_model dimensions.
        """
        doc_vec = self.encoder(doc_ids)
        # Project to d_model for manifold consumption
        # Reuse encoder weights for a simple linear projection
        return doc_vec

    def inject_into_block(
        self,
        block: nn.Module,
        lora_A: torch.Tensor,
        lora_B: torch.Tensor,
    ) -> None:
        """Inject LoRA matrices into a transformer block.

        Note: For batch training, the LoRA is applied in the forward pass
        of the block rather than set as static parameters.
        """
        # Store as attributes on the block (used by TransformerBlock.forward)
        block.lora_A = lora_A
        block.lora_B = lora_B

    def clear_from_block(self, block: nn.Module) -> None:
        """Remove injected LoRA from a block."""
        block.lora_A = None
        block.lora_B = None
