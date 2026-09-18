"""Loss functions for Bash-MANTIS.

Implements all 11 loss components:
  1. LM loss (next-token prediction)
  2. OFF feature loss (teacher-student cosine distance)
  3. Text alignment loss (paraphrase clustering)
  4. Intent bridge loss (text-to-bash latent alignment)
  5. Syntax loss (bash -n penalty)
  6. Execution loss (sandbox behavioral match, weighted by failure class)
  7. Attractor loss (fixed-point stability)
  8. Contraction loss (Jacobian spectral norm)
  9. Equivalence clustering loss (behavioral equivalence in latent space)
  10. TODO preference loss (ternary win/tie/lose)
  11. Calibration loss (typed manifold heads vs. sandbox outcomes)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class LossOutput:
    """Container for all loss components."""
    total: torch.Tensor
    lm: torch.Tensor
    off: torch.Tensor
    text_align: torch.Tensor
    intent_bridge: torch.Tensor
    syntax: torch.Tensor
    exec_loss: torch.Tensor
    attract: torch.Tensor
    contract: torch.Tensor
    equiv: torch.Tensor
    todo: torch.Tensor
    calib: torch.Tensor | None = None


class BashMantisLoss(nn.Module):
    """Combined loss function for Bash-MANTIS training."""

    def __init__(
        self,
        lambda_lm: float = 1.0,
        lambda_off: float = 0.0,
        lambda_text_align: float = 0.0,
        lambda_intent_bridge: float = 0.0,
        lambda_syntax: float = 0.0,
        lambda_exec: float = 0.0,
        lambda_attract: float = 0.0,
        lambda_contract: float = 0.0,
        lambda_equiv: float = 0.0,
        lambda_todo: float = 0.0,
        lambda_calib: float = 0.0,
        contraction_gamma: float = 0.95,
        equiv_margin: float = 1.0,
        equiv_beta: float = 0.5,
        vocab_size: int = 320,
    ):
        super().__init__()
        self.lambda_lm = lambda_lm
        self.lambda_off = lambda_off
        self.lambda_text_align = lambda_text_align
        self.lambda_intent_bridge = lambda_intent_bridge
        self.lambda_syntax = lambda_syntax
        self.lambda_exec = lambda_exec
        self.lambda_attract = lambda_attract
        self.lambda_contract = lambda_contract
        self.lambda_equiv = lambda_equiv
        self.lambda_todo = lambda_todo
        self.lambda_calib = lambda_calib
        self.contraction_gamma = contraction_gamma
        self.equiv_margin = equiv_margin
        self.equiv_beta = equiv_beta
        self.vocab_size = vocab_size

    def lm_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Standard next-token prediction loss.

        Args:
            logits: [B, T, vocab_size]
            targets: [B, T] token IDs
        """
        B, T, V = logits.shape
        return F.cross_entropy(
            logits[:, :-1].contiguous().view(-1, V),
            targets[:, 1:].contiguous().view(-1),
            ignore_index=0,  # PAD token
        )

    def off_loss(
        self,
        student_features: torch.Tensor,
        teacher_features: torch.Tensor,
    ) -> torch.Tensor:
        """OFF feature distillation loss (cosine distance).

        Args:
            student_features: [B, T, D] student hidden states
            teacher_features: [B, T, D] teacher hidden states
        """
        # Pool to sequence level
        s = student_features.mean(dim=1)
        t = teacher_features.mean(dim=1)
        return 1.0 - F.cosine_similarity(s, t, dim=-1).mean()

    def text_align_loss(
        self,
        kappa_i: torch.Tensor,
        kappa_j: torch.Tensor,
    ) -> torch.Tensor:
        """Text alignment loss: paraphrase pairs should have similar kappa.

        Args:
            kappa_i: [N, manifold_dim] kappa for paraphrase version i
            kappa_j: [N, manifold_dim] kappa for paraphrase version j
        """
        return (kappa_i - kappa_j).pow(2).mean()

    def intent_bridge_loss(
        self,
        kappa_text: torch.Tensor,
        kappa_bash: torch.Tensor,
    ) -> torch.Tensor:
        """Intent bridge loss: text instruction kappa should match bash solution kappa.

        Args:
            kappa_text: [N, manifold_dim] kappa after reading instruction
            kappa_bash: [N, manifold_dim] kappa after reading correct bash
        """
        return (kappa_text - kappa_bash).pow(2).mean()

    def syntax_loss(self, syntax_valid: torch.Tensor) -> torch.Tensor:
        """Syntax loss: penalize outputs that fail bash -n.

        Args:
            syntax_valid: [B] boolean tensor, True if bash -n passes
        """
        return (~syntax_valid).float().mean()

    def exec_loss(
        self,
        obs_pred: dict[str, torch.Tensor],
        obs_target: dict[str, torch.Tensor],
        fail_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Execution loss: compare observable behavior.

        Weighted by failure class when ``fail_weights`` is supplied, so
        the loss distinguishes kinds of wrongness instead of treating
        every mismatch alike. A quoting slip and a wrong algorithm both
        fail to match, but only one is a small correction; and a missing
        binary in the sandbox is not the model's mistake at all and gets
        weight 0.

        Args:
            obs_pred: predicted observables {exit_code, stdout_hash, ...}
            obs_target: target observables
            fail_weights: [B] per-sample weights from
                ``Observables.loss_weight``. Uniform if omitted.
        """
        device = next(iter(obs_pred.values())).device
        loss = torch.tensor(0.0, device=device)
        n = 0
        for key in obs_pred:
            if key in obs_target:
                mismatch = (obs_pred[key] != obs_target[key]).float()
                if fail_weights is not None:
                    w = fail_weights.to(device).float()
                    denom = w.sum().clamp_min(1e-8)
                    loss = loss + (mismatch * w).sum() / denom
                else:
                    loss = loss + mismatch.mean()
                n += 1
        return loss / max(n, 1)

    def calibration_loss(
        self,
        typed_heads: nn.Module,
        kappa: torch.Tensor,
        repair_target: torch.Tensor | None = None,
        safety_target: torch.Tensor | None = None,
        completion_target: torch.Tensor | None = None,
        completion_idx: torch.Tensor | None = None,
        repair_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Calibration loss for the typed manifold heads.

        Supervises kappa's declared dimensions against outcomes observed
        in the sandbox rather than against human preference, which is
        what makes the resulting probabilities safe to threshold on.

        The ``*_idx`` tensors select which samples carry each label —
        repair supervision only exists for fix_bash tasks, and completion
        supervision is dropped for blameless environment failures.
        """
        k_completion = kappa[completion_idx] if completion_idx is not None else kappa
        k_repair = kappa[repair_idx] if repair_idx is not None else kappa

        total = torch.zeros((), device=kappa.device)
        if completion_target is not None and completion_target.numel():
            total = total + typed_heads.completion.loss(k_completion, completion_target)
        if repair_target is not None and repair_target.numel():
            total = total + typed_heads.repair.loss(k_repair, repair_target)
        if safety_target is not None and safety_target.numel():
            total = total + typed_heads.safety.loss(kappa, safety_target)
        return total

    def attractor_loss(
        self,
        kappa_star: torch.Tensor,
        manifold: nn.Module,
        h_t: torch.Tensor,
        p_bar: torch.Tensor,
        m_t: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Attractor loss: correct terminal states should be fixed points.

        ||F(kappa*, u, m) - kappa*||^2 should be small.
        """
        kappa_next = manifold.step(kappa_star, h_t, p_bar, m_t)
        return (kappa_next - kappa_star).pow(2).mean()

    def contraction_loss(
        self,
        manifold: nn.Module,
        kappa_star: torch.Tensor,
        h_t: torch.Tensor,
        p_bar: torch.Tensor,
        m_t: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Contraction loss: max singular value of Jacobian should be < gamma."""
        sigma_max = manifold.compute_jacobian_norm(kappa_star, h_t, p_bar, m_t)
        return F.relu(sigma_max - self.contraction_gamma).pow(2)

    def equiv_loss(
        self,
        kappa_equiv_pairs: list[tuple[torch.Tensor, torch.Tensor]],
        kappa_neg_pairs: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        """Equivalence clustering loss.

        Equivalent bash outputs -> close in latent space.
        Non-equivalent outputs -> separated by margin.
        """
        loss = torch.tensor(0.0)
        if kappa_equiv_pairs:
            device = kappa_equiv_pairs[0][0].device
            loss = loss.to(device)

        # Pull equivalent pairs together
        for ki, kj in kappa_equiv_pairs:
            loss = loss + (ki - kj).pow(2).sum()
        if kappa_equiv_pairs:
            loss = loss / len(kappa_equiv_pairs)

        # Push non-equivalent pairs apart
        neg_loss = torch.tensor(0.0, device=loss.device)
        for ki, kj in kappa_neg_pairs:
            dist = (ki - kj).pow(2).sum().sqrt()
            neg_loss = neg_loss + F.relu(self.equiv_margin - dist).pow(2)
        if kappa_neg_pairs:
            neg_loss = neg_loss / len(kappa_neg_pairs)

        return loss + self.equiv_beta * neg_loss

    def todo_loss(
        self,
        preference_head: nn.Module,
        rep_a: torch.Tensor,
        rep_b: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """TODO preference loss: ternary win/tie/lose.

        Args:
            preference_head: PreferenceHead module
            rep_a: [B, d_model] candidate A representation
            rep_b: [B, d_model] candidate B representation
            labels: [B] with values in {0=win, 1=tie, 2=lose}
        """
        return preference_head.compute_loss(rep_a, rep_b, labels)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        teacher_features: torch.Tensor | None = None,
        student_features: torch.Tensor | None = None,
        kappa: torch.Tensor | None = None,
        manifold: nn.Module | None = None,
        h_t: torch.Tensor | None = None,
        p_bar: torch.Tensor | None = None,
        m_t: torch.Tensor | None = None,
        syntax_valid: torch.Tensor | None = None,
        obs_pred: dict | None = None,
        obs_target: dict | None = None,
        kappa_paraphrase_i: torch.Tensor | None = None,
        kappa_paraphrase_j: torch.Tensor | None = None,
        kappa_text: torch.Tensor | None = None,
        kappa_bash: torch.Tensor | None = None,
        kappa_equiv_pairs: list | None = None,
        kappa_neg_pairs: list | None = None,
        preference_head: nn.Module | None = None,
        pref_rep_a: torch.Tensor | None = None,
        pref_rep_b: torch.Tensor | None = None,
        pref_labels: torch.Tensor | None = None,
        fail_weights: torch.Tensor | None = None,
        typed_heads: nn.Module | None = None,
        repair_target: torch.Tensor | None = None,
        safety_target: torch.Tensor | None = None,
        completion_target: torch.Tensor | None = None,
        completion_idx: torch.Tensor | None = None,
        repair_idx: torch.Tensor | None = None,
    ) -> LossOutput:
        """Compute all active loss components and return weighted sum."""
        device = logits.device
        zero = torch.tensor(0.0, device=device)

        # 1. LM loss (always active)
        l_lm = self.lm_loss(logits, targets)

        # 2. OFF loss
        l_off = zero
        if self.lambda_off > 0 and teacher_features is not None and student_features is not None:
            l_off = self.off_loss(student_features, teacher_features)

        # 3. Text alignment
        l_text_align = zero
        if self.lambda_text_align > 0 and kappa_paraphrase_i is not None:
            l_text_align = self.text_align_loss(kappa_paraphrase_i, kappa_paraphrase_j)

        # 4. Intent bridge
        l_intent_bridge = zero
        if self.lambda_intent_bridge > 0 and kappa_text is not None:
            l_intent_bridge = self.intent_bridge_loss(kappa_text, kappa_bash)

        # 5. Syntax
        l_syntax = zero
        if self.lambda_syntax > 0 and syntax_valid is not None:
            l_syntax = self.syntax_loss(syntax_valid)

        # 6. Execution (failure-class weighted)
        l_exec = zero
        if self.lambda_exec > 0 and obs_pred is not None:
            l_exec = self.exec_loss(obs_pred, obs_target, fail_weights)

        # 7. Attractor
        l_attract = zero
        if self.lambda_attract > 0 and kappa is not None and manifold is not None:
            l_attract = self.attractor_loss(kappa, manifold, h_t, p_bar, m_t)

        # 8. Contraction
        l_contract = zero
        if self.lambda_contract > 0 and manifold is not None and kappa is not None:
            l_contract = self.contraction_loss(manifold, kappa, h_t, p_bar, m_t)

        # 9. Equivalence
        l_equiv = zero
        if self.lambda_equiv > 0 and kappa_equiv_pairs is not None:
            l_equiv = self.equiv_loss(kappa_equiv_pairs, kappa_neg_pairs or [])

        # 10. TODO preference
        l_todo = zero
        if self.lambda_todo > 0 and preference_head is not None and pref_rep_a is not None:
            l_todo = self.todo_loss(preference_head, pref_rep_a, pref_rep_b, pref_labels)

        # 11. Calibration of the typed manifold heads
        l_calib = zero
        if self.lambda_calib > 0 and typed_heads is not None and kappa is not None:
            l_calib = self.calibration_loss(
                typed_heads,
                kappa,
                repair_target=repair_target,
                safety_target=safety_target,
                completion_target=completion_target,
                completion_idx=completion_idx,
                repair_idx=repair_idx,
            )

        # Weighted sum
        total = (
            self.lambda_lm * l_lm
            + self.lambda_off * l_off
            + self.lambda_text_align * l_text_align
            + self.lambda_intent_bridge * l_intent_bridge
            + self.lambda_syntax * l_syntax
            + self.lambda_exec * l_exec
            + self.lambda_attract * l_attract
            + self.lambda_contract * l_contract
            + self.lambda_equiv * l_equiv
            + self.lambda_todo * l_todo
            + self.lambda_calib * l_calib
        )

        return LossOutput(
            total=total,
            lm=l_lm,
            off=l_off,
            text_align=l_text_align,
            intent_bridge=l_intent_bridge,
            syntax=l_syntax,
            exec_loss=l_exec,
            attract=l_attract,
            contract=l_contract,
            equiv=l_equiv,
            todo=l_todo,
            calib=l_calib,
        )

    @classmethod
    def from_config(cls, config) -> "BashMantisLoss":
        """Create loss from training config."""
        tc = config.training
        return cls(
            lambda_lm=tc.lambda_lm,
            lambda_off=tc.lambda_off,
            lambda_text_align=tc.lambda_text_align,
            lambda_intent_bridge=tc.lambda_intent_bridge,
            lambda_syntax=tc.lambda_syntax,
            lambda_exec=tc.lambda_exec,
            lambda_attract=tc.lambda_attract,
            lambda_contract=tc.lambda_contract,
            lambda_equiv=tc.lambda_equiv,
            lambda_todo=tc.lambda_todo,
            lambda_calib=getattr(tc, "lambda_calib", 0.0),
            contraction_gamma=tc.contraction_gamma,
            equiv_margin=tc.equiv_margin,
            equiv_beta=tc.equiv_beta,
            vocab_size=config.model.vocab_size,
        )
