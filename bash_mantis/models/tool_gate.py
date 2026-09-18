"""Tool-call gate: should this shell command be run, and by whom?

Bash-MANTIS is far too small to generate tool calls, and structured
call syntax out of a byte-level decoder is exactly the failure mode a
typed decision model exists to avoid. But the judgment *around* a tool
call -- is this safe to execute unattended, does it need a sandbox
first, does a human have to look at it -- is a small, closed-form
decision with real ground truth behind it. That is the shape this model
can fill, at roughly the cost of one matmul.

The gate answers three questions about a proposed command:

    route       Choice over {auto, sandbox, confirm, reject}
    safety_risk Noul, P(the command is destructive)
    confidence  mass on the chosen route

and the caller thresholds on confidence:

    d = gate.decide(kappa)
    if d.confidence < 0.8:
        escalate()                 # the model is not sure; do not guess

Routes, lowest to highest intervention:

    AUTO     run it; no destructive pattern, and it behaved
    SANDBOX  run it somewhere disposable first; it failed or is untested
    CONFIRM  a human approves before it runs; destructive but plausible
    REJECT   do not run it

The ordering above is intuitive but deliberately NOT modeled as ordered:
CONFIRM is not "between" SANDBOX and REJECT in any sense a cumulative
link would capture -- it is a different kind of action, involving a
different actor. Hence Choice rather than Score.

Fail-closed is not free here. A gate that says AUTO on a destructive
command is far more costly than one that says CONFIRM on a harmless
one, so `decide` takes a conservative floor: below the confidence
threshold it escalates rather than trusting its own argmax.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from bash_mantis.eval.sandbox_exec import FailClass, Observables
from bash_mantis.models.typed_heads import (
    KAPPA_SAFETY_MODE,
    ChoiceHead,
    NoulHead,
)


# Route options, in order of increasing intervention.
AUTO = 0
SANDBOX = 1
CONFIRM = 2
REJECT = 3

ROUTE_NAMES = ("auto", "sandbox", "confirm", "reject")

#: Routes that must never be reached by a low-confidence guess.
_GUARDED = frozenset({AUTO})

#: Where an unconfident gate falls back to. Not REJECT: refusing
#: everything the model is unsure about makes it useless. A human
#: looking at it is the honest answer to "I don't know".
ESCALATION_ROUTE = CONFIRM


@dataclass
class GateDecision:
    """One gate decision, per command."""
    route: int
    confidence: float
    safety_risk: float
    escalated: bool = False

    @property
    def name(self) -> str:
        return ROUTE_NAMES[self.route]

    @property
    def may_auto_run(self) -> bool:
        return self.route == AUTO

    def __repr__(self) -> str:
        tail = " (escalated)" if self.escalated else ""
        return (f"<{self.name} conf={self.confidence:.2f} "
                f"risk={self.safety_risk:.2f}{tail}>")


def route_label(script_is_destructive: bool, obs: Observables,
                matched: bool) -> int | None:
    """Ground-truth route for a command whose outcome is known.

    Returns None when the outcome says nothing about the command -- an
    environment failure means the sandbox lacked a binary, which is not
    evidence about whether the command should have been trusted.

    Destructive commands are labelled CONFIRM rather than REJECT even
    when they ran fine: `rm -rf build/` is a normal thing to want, and
    training the gate to refuse it outright would teach it to refuse a
    large slice of legitimate shell work. REJECT is reserved for
    destructive *and* broken -- a command that would do damage and does
    not even do what was asked.
    """
    if obs.blameless:
        return None

    if script_is_destructive:
        return CONFIRM if (matched or obs.fail_class == FailClass.OK) else REJECT

    if matched:
        return AUTO
    return SANDBOX


class ToolCallGate(nn.Module):
    """Routing gate over the manifold state.

    Shares kappa[10] with the safety head in TypedManifoldHeads: the
    same dimension the calibration loss already trains against
    destructive-pattern ground truth. The gate reuses that signal rather
    than learning a second, possibly disagreeing, notion of danger.
    """

    N_ROUTES = len(ROUTE_NAMES)

    def __init__(self, manifold_dim: int = 12,
                 confidence_threshold: float = 0.8):
        super().__init__()
        self.confidence_threshold = confidence_threshold
        self.route = ChoiceHead(self.N_ROUTES, manifold_dim=manifold_dim)
        self.safety = NoulHead(KAPPA_SAFETY_MODE)

    def forward(self, kappa: torch.Tensor) -> dict[str, torch.Tensor]:
        """Raw gate outputs, before any thresholding."""
        return {
            "route": self.route(kappa),
            "route_probs": self.route.probabilities(kappa),
            "route_confidence": self.route.confidence(kappa),
            "safety_risk": self.safety(kappa),
        }

    @torch.no_grad()
    def decide(self, kappa: torch.Tensor) -> list[GateDecision]:
        """Gate decisions with the conservative floor applied.

        A guarded route (AUTO) reached below the confidence threshold is
        escalated. Higher-intervention routes are left alone: being
        unsure is not a reason to downgrade a command *toward* running
        it unattended.
        """
        out = self.forward(kappa)
        routes = out["route"].tolist()
        confs = out["route_confidence"].tolist()
        risks = out["safety_risk"].tolist()

        decisions = []
        for r, c, risk in zip(routes, confs, risks):
            escalated = r in _GUARDED and c < self.confidence_threshold
            decisions.append(GateDecision(
                route=ESCALATION_ROUTE if escalated else r,
                confidence=c, safety_risk=risk, escalated=escalated,
            ))
        return decisions

    def loss(self, kappa: torch.Tensor, route_target: torch.Tensor,
             safety_target: torch.Tensor | None = None) -> torch.Tensor:
        """Routing cross-entropy, plus safety BCE when labels exist."""
        total = self.route.loss(kappa, route_target)
        if safety_target is not None and safety_target.numel():
            total = total + self.safety.loss(kappa, safety_target)
        return total


@torch.no_grad()
def gate_risk_report(decisions: list[GateDecision],
                     truth: list[int | None]) -> dict[str, float]:
    """Score a gate against known-correct routes.

    Plain accuracy is the wrong headline for a safety gate, because the
    errors are not symmetric. What matters is `unsafe_auto`: how often
    the gate waved through something that should have been stopped.
    A gate with worse accuracy and zero unsafe_auto is the better gate.
    """
    pairs = [(d, t) for d, t in zip(decisions, truth) if t is not None]
    if not pairs:
        return {"n": 0}

    n = len(pairs)
    correct = sum(d.route == t for d, t in pairs)
    unsafe = sum(d.route == AUTO and t in (CONFIRM, REJECT) for d, t in pairs)
    overcautious = sum(d.route != AUTO and t == AUTO for d, t in pairs)
    escalated = sum(d.escalated for d, _ in pairs)

    return {
        "n": n,
        "accuracy": correct / n,
        "unsafe_auto": unsafe / n,       # the number that must be ~0
        "overcautious": overcautious / n,
        "escalated": escalated / n,
    }
