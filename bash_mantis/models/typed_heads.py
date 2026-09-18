"""Calibrated typed decision heads over the manifold state.

The manifold docstring assigns a meaning to each of kappa's 12 dimensions
("repair confidence", "safety mode", "completion confidence", ...), but
nothing in the loss ever enforced those meanings — they were aspirational
labels on unconstrained latents.

This module makes three of them real by reading a *single declared
dimension* through a monotone link and supervising it against ground
truth from the sandbox.  Because each head reads one scalar, the only way
to drive its loss down is for that dimension to actually carry that
quantity; there is no room to smuggle the signal in elsewhere.

Two primitives, following the decision-model convention:

  Noul   yes/no.  Returns a probability in [0, 1].  The number *is* the
         belief, so there is no separate confidence field.
  Score  position on an ordered scale.  Returns a fractional position in
         [0, levels-1] — the probability-weighted mean, NOT an argmax
         index.  A score of 1.6 on a 3-level scale means the mass sits
         between levels 1 and 2, leaning toward 2.

Scores use an ordinal (cumulative-link) parameterization so the levels
stay ordered by construction: a higher latent can never make a lower
level more likely.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# Declared kappa dimensions (see manifold.ManifoldState docstring).
KAPPA_REPAIR_CONFIDENCE = 7
KAPPA_SAFETY_MODE = 10
KAPPA_COMPLETION_CONFIDENCE = 11


class NoulHead(nn.Module):
    """Yes/no decision read from one manifold dimension.

    p = sigmoid(scale * kappa[dim] + bias)

    The affine is deliberately minimal: with a single scalar input and a
    monotone link, driving the loss down *requires* kappa[dim] to track
    the outcome.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.scale = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def logit(self, kappa: torch.Tensor) -> torch.Tensor:
        """Return the raw logit. Args: kappa [B, manifold_dim] -> [B]."""
        return self.scale * kappa[:, self.dim] + self.bias

    def forward(self, kappa: torch.Tensor) -> torch.Tensor:
        """Return P(yes). Args: kappa [B, manifold_dim] -> [B]."""
        return torch.sigmoid(self.logit(kappa))

    def loss(self, kappa: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Binary cross-entropy against observed outcomes.

        Args:
            kappa: [B, manifold_dim]
            targets: [B] float in {0., 1.}
        """
        return F.binary_cross_entropy_with_logits(
            self.logit(kappa), targets.float()
        )


class ScoreHead(nn.Module):
    """Ordered-level decision read from one manifold dimension.

    Uses a cumulative link:  P(Y > k) = sigmoid(z - theta_k), with
    thresholds kept strictly increasing via a softplus-cumsum
    parameterization. This guarantees the level ordering is respected no
    matter what the optimizer does to the parameters.
    """

    def __init__(self, dim: int, levels: int = 3):
        super().__init__()
        if levels < 2:
            raise ValueError("a Score needs at least 2 levels")
        self.dim = dim
        self.levels = levels
        self.scale = nn.Parameter(torch.ones(1))
        # theta_0 is free; each subsequent threshold is strictly greater.
        self.theta_0 = nn.Parameter(torch.zeros(1))
        self.theta_deltas = nn.Parameter(torch.zeros(levels - 2)) if levels > 2 else None

    def _thresholds(self) -> torch.Tensor:
        """Strictly increasing thresholds, shape [levels-1]."""
        if self.theta_deltas is None:
            return self.theta_0
        steps = F.softplus(self.theta_deltas) + 1e-4
        return torch.cat([self.theta_0, self.theta_0 + torch.cumsum(steps, dim=0)])

    def probabilities(self, kappa: torch.Tensor) -> torch.Tensor:
        """Per-level probabilities. Args: kappa [B, D] -> [B, levels]."""
        z = (self.scale * kappa[:, self.dim]).unsqueeze(-1)   # [B, 1]
        cum = torch.sigmoid(z - self._thresholds().unsqueeze(0))  # [B, levels-1]

        first = 1.0 - cum[:, :1]                 # P(Y = 0)
        last = cum[:, -1:]                       # P(Y = levels-1)
        if self.levels == 2:
            return torch.cat([first, last], dim=-1)
        middle = cum[:, :-1] - cum[:, 1:]        # P(Y = 1 .. levels-2)
        return torch.cat([first, middle, last], dim=-1)

    def forward(self, kappa: torch.Tensor) -> torch.Tensor:
        """Fractional score in [0, levels-1] — probability-weighted mean.

        Deliberately not an argmax: the between-level position carries
        the model's uncertainty and is lost by rounding.
        """
        probs = self.probabilities(kappa)
        idx = torch.arange(self.levels, device=kappa.device, dtype=probs.dtype)
        return (probs * idx).sum(dim=-1)

    def confidence(self, kappa: torch.Tensor) -> torch.Tensor:
        """Probability mass on the most likely level. -> [B]."""
        return self.probabilities(kappa).max(dim=-1).values

    def loss(self, kappa: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Negative log-likelihood of the observed level.

        Args:
            kappa: [B, manifold_dim]
            targets: [B] long in [0, levels-1]
        """
        probs = self.probabilities(kappa).clamp_min(1e-8)
        return F.nll_loss(probs.log(), targets.long())


class TypedManifoldHeads(nn.Module):
    """The three manifold dimensions that have sandbox ground truth.

    Outcome labels come from execution, not from human preference, so a
    low loss here means the probabilities are calibrated against what
    actually happens — which is what makes thresholding on them
    (``if completion_confidence < tau: sample more``) sound rather than
    superstitious.
    """

    #: Levels for the completion Score.
    COMPLETION_LEVELS = 3   # 0 = syntax fail, 1 = ran but wrong, 2 = matched

    def __init__(self):
        super().__init__()
        self.repair = NoulHead(KAPPA_REPAIR_CONFIDENCE)
        self.safety = NoulHead(KAPPA_SAFETY_MODE)
        self.completion = ScoreHead(
            KAPPA_COMPLETION_CONFIDENCE, levels=self.COMPLETION_LEVELS
        )

    def forward(self, kappa: torch.Tensor) -> dict[str, torch.Tensor]:
        """Read all typed answers from kappa.

        Returns a dict with:
            repair_confidence:     [B] P(this repair fixes the script)
            safety_risk:           [B] P(this script is destructive)
            completion_score:      [B] fractional position in [0, 2]
            completion_confidence: [B] mass on the most likely level
        """
        return {
            "repair_confidence": self.repair(kappa),
            "safety_risk": self.safety(kappa),
            "completion_score": self.completion(kappa),
            "completion_confidence": self.completion.confidence(kappa),
        }

    def calibration_loss(
        self,
        kappa: torch.Tensor,
        repair_target: torch.Tensor | None = None,
        safety_target: torch.Tensor | None = None,
        completion_target: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Sum of per-head losses over whichever targets are supplied.

        Targets are optional because not every batch carries every label:
        repair supervision only exists for fix_bash tasks, and completion
        supervision only when the sandbox actually ran.
        """
        device = kappa.device
        total = torch.zeros((), device=device)

        if repair_target is not None and repair_target.numel():
            total = total + self.repair.loss(kappa, repair_target)
        if safety_target is not None and safety_target.numel():
            total = total + self.safety.loss(kappa, safety_target)
        if completion_target is not None and completion_target.numel():
            total = total + self.completion.loss(kappa, completion_target)

        return total


@torch.no_grad()
def expected_calibration_error(
    probs: torch.Tensor,
    outcomes: torch.Tensor,
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error for a set of binary predictions.

    Buckets predictions by confidence and measures the gap between stated
    confidence and observed accuracy in each bucket. This is the number
    that says whether "0.8 confident" really means right 80% of the time
    — the claim the whole typed-head design rests on. Lower is better;
    0 is perfectly calibrated.

    Args:
        probs: [N] predicted probabilities in [0, 1]
        outcomes: [N] observed binary outcomes
    """
    probs = probs.flatten().float()
    outcomes = outcomes.flatten().float()
    n = probs.numel()
    if n == 0:
        return 0.0

    edges = torch.linspace(0, 1, n_bins + 1, device=probs.device)
    ece = torch.zeros((), device=probs.device)

    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        # Include the left edge on the first bin so 0.0 is counted.
        in_bin = (probs > lo) & (probs <= hi) if i else (probs >= lo) & (probs <= hi)
        count = in_bin.sum()
        if count == 0:
            continue
        acc = outcomes[in_bin].mean()
        conf = probs[in_bin].mean()
        ece = ece + (count / n) * (acc - conf).abs()

    return ece.item()


class ChoiceHead(nn.Module):
    """One-of-N decision read from the whole manifold state.

    The Noul and Score heads each read a single declared dimension,
    because a scalar through a monotone link is enough to carry a yes/no
    or an ordered position. A Choice over N options is not orderable and
    does not fit in one scalar, so this head projects all of kappa.

    That is a real trade-off: it gives up the "this dimension means this
    thing" guarantee the other heads have. It is worth it only because
    routing is a genuinely N-way decision with no natural ordering --
    "send to a human" is not between "auto-execute" and "reject".
    """

    def __init__(self, n_options: int, manifold_dim: int = 12):
        super().__init__()
        if n_options < 2:
            raise ValueError("a Choice needs at least 2 options")
        self.n_options = n_options
        self.proj = nn.Linear(manifold_dim, n_options)

    def logits(self, kappa: torch.Tensor) -> torch.Tensor:
        """Raw per-option logits. Args: kappa [B, D] -> [B, n_options]."""
        return self.proj(kappa)

    def probabilities(self, kappa: torch.Tensor) -> torch.Tensor:
        """Per-option probabilities. -> [B, n_options]."""
        return F.softmax(self.logits(kappa), dim=-1)

    def forward(self, kappa: torch.Tensor) -> torch.Tensor:
        """Most likely option index. -> [B] long.

        Unlike a Score, an argmax *is* the right reduction here: the
        options are unordered, so a weighted mean would be meaningless.
        """
        return self.logits(kappa).argmax(dim=-1)

    def confidence(self, kappa: torch.Tensor) -> torch.Tensor:
        """Probability mass on the chosen option. -> [B]."""
        return self.probabilities(kappa).max(dim=-1).values

    def loss(self, kappa: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Cross-entropy against the observed routing decision."""
        return F.cross_entropy(self.logits(kappa), targets.long())
