"""Ternary linear layer with DLT-style learned reconstruction.

Implements {-1, 0, +1} quantization of weights with:
- Groupwise learned scale and shift parameters
- Straight-through estimator for gradient flow
- Optional full-precision island within each layer
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class TernaryQuantize(torch.autograd.Function):
    """Ternary quantization with straight-through estimator."""

    @staticmethod
    def forward(ctx, weight: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor):
        # Normalize weight within each group
        w_shifted = weight - shift
        w_normalized = w_shifted / (scale + 1e-8)
        # Quantize to {-1, 0, +1}
        ternary = torch.sign(w_normalized) * (w_normalized.abs() > 0.5).float()
        # Reconstruct with learned scale/shift
        reconstructed = ternary * scale + shift
        ctx.save_for_backward(weight, scale, shift)
        return reconstructed

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through estimator: pass gradients through
        weight, scale, shift = ctx.saved_tensors
        grad_weight = grad_output.clone()
        # Gradient for scale: sum of (ternary * grad_output) per group
        grad_scale = (grad_output * torch.sign(weight - shift)).sum(dim=-1, keepdim=True)
        grad_shift = grad_output.sum(dim=-1, keepdim=True)
        return grad_weight, grad_scale, grad_shift


class TernaryLinear(nn.Module):
    """Linear layer with ternary weight quantization.

    Supports:
    - DLT-style groupwise scale/shift for learned reconstruction
    - A small full-precision "island" that stays in fp32
    - Toggling between dense (full-precision) and ternary modes
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        ternary: bool = False,
        group_size: int = 16,
        precision_island_size: int = 16,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ternary = ternary
        self.group_size = group_size
        self.precision_island_size = precision_island_size

        # Main weight
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        # DLT groupwise scale and shift
        n_groups = max(1, out_features * in_features // group_size)
        self.scale = nn.Parameter(torch.ones(n_groups, 1))
        self.shift = nn.Parameter(torch.zeros(n_groups, 1))

        # Precision island: a small full-precision linear bypass
        if precision_island_size > 0:
            self.island = nn.Linear(
                min(precision_island_size, in_features),
                min(precision_island_size, out_features),
                bias=False,
            )
        else:
            self.island = None

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_features
            bound = 1.0 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.ternary:
            # Reshape weight for groupwise quantization
            w_flat = self.weight.view(-1)
            n = w_flat.numel()
            gs = self.group_size
            # Pad if needed
            pad_n = (gs - n % gs) % gs
            if pad_n > 0:
                w_flat = F.pad(w_flat, (0, pad_n))
            w_grouped = w_flat.view(-1, gs)
            # Match scale/shift to number of groups
            n_groups = w_grouped.shape[0]
            scale = self.scale[:n_groups]
            shift = self.shift[:n_groups]
            # Quantize
            w_q = TernaryQuantize.apply(w_grouped, scale, shift)
            w_q = w_q.view(-1)[:n].view_as(self.weight)
            out = F.linear(x, w_q, self.bias)
        else:
            out = F.linear(x, self.weight, self.bias)

        # Add precision island contribution
        if self.island is not None:
            island_in = x[..., : self.island.in_features]
            island_out = self.island(island_in)
            out[..., : self.island.out_features] = (
                out[..., : self.island.out_features] + island_out
            )

        return out

    def enable_ternary(self):
        self.ternary = True

    def disable_ternary(self):
        self.ternary = False
