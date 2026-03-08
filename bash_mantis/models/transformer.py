"""Transformer backbone for Bash-MANTIS.

A tiny 4-layer transformer with:
- Multi-head self-attention
- FFN with optional ternary linear layers
- RMSNorm (kept in full precision)
- Rotary position embeddings
- Optional LoRA injection point
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from bash_mantis.models.ternary_linear import TernaryLinear


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (full precision)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding."""

    def __init__(self, dim: int, max_seq_len: int = 512, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, inv_freq)
        self.register_buffer("cos_cached", freqs.cos())
        self.register_buffer("sin_cached", freqs.sin())

    def forward(self, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary embeddings to input tensor.

    Handles odd dimensions by applying rotary only to the even portion
    and passing the remainder through unchanged.
    """
    d = x.shape[-1]
    rot_dim = (d // 2) * 2  # largest even number <= d
    x_rot = x[..., :rot_dim]
    x_pass = x[..., rot_dim:]  # empty if d is even
    x1, x2 = x_rot[..., : rot_dim // 2], x_rot[..., rot_dim // 2 :]
    cos = cos[:x.shape[-2], :x1.shape[-1]]
    sin = sin[:x.shape[-2], :x1.shape[-1]]
    x_rot = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
    if x_pass.shape[-1] > 0:
        return torch.cat([x_rot, x_pass], dim=-1)
    return x_rot


class Attention(nn.Module):
    """Multi-head self-attention with optional ternary weights."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ternary: bool = False,
        precision_island_size: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        assert d_model % n_heads == 0

        Linear = TernaryLinear if True else nn.Linear
        kwargs = {"ternary": ternary, "precision_island_size": precision_island_size}

        self.q_proj = Linear(d_model, d_model, **kwargs)
        self.k_proj = Linear(d_model, d_model, **kwargs)
        self.v_proj = Linear(d_model, d_model, **kwargs)
        self.o_proj = Linear(d_model, d_model, **kwargs)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Apply rotary embeddings
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        # Scaled dot-product attention
        scale = self.head_dim ** 0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) / scale

        if mask is not None:
            attn = attn.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(out)


class FeedForward(nn.Module):
    """FFN with optional ternary weights and GELU activation."""

    def __init__(
        self,
        d_model: int,
        ffn_dim: int,
        ternary: bool = False,
        precision_island_size: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.w1 = TernaryLinear(d_model, ffn_dim, ternary=ternary,
                                precision_island_size=precision_island_size)
        self.w2 = TernaryLinear(ffn_dim, d_model, ternary=ternary,
                                precision_island_size=precision_island_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.gelu(self.w1(x))))


class TransformerBlock(nn.Module):
    """Single transformer block with pre-norm."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ffn_dim: int,
        ternary: bool = False,
        precision_island_size: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.attn = Attention(d_model, n_heads, ternary, precision_island_size, dropout)
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = FeedForward(d_model, ffn_dim, ternary, precision_island_size, dropout)

        # LoRA injection hooks (filled by DocToLoRA)
        self.lora_A: nn.Parameter | None = None
        self.lora_B: nn.Parameter | None = None

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Pre-norm attention
        x = x + self.attn(self.attn_norm(x), cos, sin, mask)

        # Pre-norm FFN
        h = self.ffn_norm(x)
        ffn_out = self.ffn(h)

        # Apply LoRA if injected
        if self.lora_A is not None and self.lora_B is not None:
            lora_out = h @ self.lora_A @ self.lora_B
            ffn_out = ffn_out + lora_out

        x = x + ffn_out
        return x


class TransformerBackbone(nn.Module):
    """Full transformer backbone for Bash-MANTIS."""

    def __init__(
        self,
        vocab_size: int = 320,
        d_model: int = 108,
        n_heads: int = 4,
        n_layers: int = 4,
        ffn_dim: int = 256,
        ctx_len: int = 512,
        ternary: bool = False,
        precision_island_size: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.ctx_len = ctx_len

        # Token embedding (full precision)
        self.tok_emb = nn.Embedding(vocab_size, d_model)

        # Rotary position embedding
        self.rotary = RotaryEmbedding(d_model // n_heads, ctx_len)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, ffn_dim, ternary, precision_island_size, dropout)
            for _ in range(n_layers)
        ])

        # Final norm (full precision)
        self.final_norm = RMSNorm(d_model)

        # Output head (full precision)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.tok_emb.weight

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.tok_emb.weight, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear) and module is not self.lm_head:
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Returns:
            logits: [B, T, vocab_size]
            hidden: [B, T, d_model] - last hidden states for manifold/workspace
        """
        B, T = input_ids.shape
        x = self.tok_emb(input_ids)

        # Causal mask
        if mask is None:
            mask = torch.tril(torch.ones(T, T, device=x.device)).unsqueeze(0).unsqueeze(0)

        cos, sin = self.rotary(T)

        for block in self.blocks:
            x = block(x, cos, sin, mask)

        hidden = self.final_norm(x)
        logits = self.lm_head(hidden)

        return logits, hidden

    def get_hidden(self, input_ids: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Get only the hidden states (no logit computation)."""
        _, hidden = self.forward(input_ids, mask)
        return hidden

    def enable_ternary(self):
        """Enable ternary quantization on all applicable layers."""
        for module in self.modules():
            if isinstance(module, TernaryLinear):
                module.enable_ternary()

    def disable_ternary(self):
        """Disable ternary quantization."""
        for module in self.modules():
            if isinstance(module, TernaryLinear):
                module.disable_ternary()
