"""Tests for the tool-call gate.

The gate's failure modes are asymmetric: waving through a destructive
command is far worse than over-escalating a harmless one. Most of these
tests are about that asymmetry rather than about accuracy.

Run: pytest tests/test_tool_gate.py -v
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from bash_mantis.eval.sandbox_exec import FailClass, Observables  # noqa: E402
from bash_mantis.models.tool_gate import (  # noqa: E402
    AUTO,
    CONFIRM,
    ESCALATION_ROUTE,
    REJECT,
    ROUTE_NAMES,
    SANDBOX,
    GateDecision,
    ToolCallGate,
    gate_risk_report,
    route_label,
)
from bash_mantis.models.typed_heads import ChoiceHead  # noqa: E402

MANIFOLD_DIM = 12


def _obs(fail_class=FailClass.OK, exit_code=0):
    return Observables(exit_code, "", "", {}, fail_class=fail_class)


@pytest.fixture
def kappa():
    torch.manual_seed(0)
    return torch.randn(16, MANIFOLD_DIM)


# --------------------------------------------------------------------------
# ChoiceHead
# --------------------------------------------------------------------------

def test_choice_probabilities_are_a_distribution(kappa):
    head = ChoiceHead(4, manifold_dim=MANIFOLD_DIM)
    p = head.probabilities(kappa)
    assert p.shape == (16, 4)
    assert (p >= 0).all()
    assert torch.allclose(p.sum(-1), torch.ones(16), atol=1e-5)


def test_choice_returns_argmax_not_a_mean(kappa):
    """Options are unordered; a weighted mean would be meaningless."""
    head = ChoiceHead(4, manifold_dim=MANIFOLD_DIM)
    chosen = head(kappa)
    assert chosen.dtype == torch.long
    assert torch.equal(chosen, head.probabilities(kappa).argmax(-1))


def test_choice_confidence_is_chosen_option_mass(kappa):
    head = ChoiceHead(4, manifold_dim=MANIFOLD_DIM)
    assert torch.allclose(
        head.confidence(kappa), head.probabilities(kappa).max(-1).values)
    assert (head.confidence(kappa) >= 0.25 - 1e-6).all()


def test_choice_rejects_degenerate_option_count():
    with pytest.raises(ValueError):
        ChoiceHead(1, manifold_dim=MANIFOLD_DIM)


# --------------------------------------------------------------------------
# Ground-truth route derivation
# --------------------------------------------------------------------------

def test_safe_and_matching_is_auto():
    assert route_label(False, _obs(), matched=True) == AUTO


def test_safe_but_failing_needs_a_sandbox():
    assert route_label(False, _obs(FailClass.SYNTAX, 2), matched=False) == SANDBOX
    assert route_label(False, _obs(FailClass.SEMANTIC), matched=False) == SANDBOX


def test_destructive_but_working_asks_a_human_rather_than_refusing():
    """`rm -rf build/` is legitimate; refusing it outright is useless."""
    assert route_label(True, _obs(), matched=True) == CONFIRM
    assert route_label(True, _obs(), matched=False) == CONFIRM


def test_destructive_and_broken_is_rejected():
    assert route_label(True, _obs(FailClass.SYNTAX, 2), matched=False) == REJECT


def test_environment_failure_yields_no_label():
    """A missing binary says nothing about whether to trust the command."""
    assert route_label(False, _obs(FailClass.ENV, 127), matched=False) is None
    assert route_label(True, _obs(FailClass.ENV, 127), matched=False) is None


def test_destructive_never_labelled_auto():
    """The property that matters: no destructive command trains toward AUTO."""
    for fc in (FailClass.OK, FailClass.SYNTAX, FailClass.RUNTIME,
               FailClass.SEMANTIC, FailClass.TIMEOUT):
        for matched in (True, False):
            label = route_label(True, _obs(fc), matched)
            assert label != AUTO, f"destructive -> AUTO via {fc}/{matched}"


# --------------------------------------------------------------------------
# Gate decisions and the conservative floor
# --------------------------------------------------------------------------

def test_gate_forward_shapes(kappa):
    gate = ToolCallGate(manifold_dim=MANIFOLD_DIM)
    out = gate(kappa)
    assert out["route"].shape == (16,)
    assert out["route_probs"].shape == (16, gate.N_ROUTES)
    assert ((out["safety_risk"] >= 0) & (out["safety_risk"] <= 1)).all()


def test_untrained_gate_escalates_rather_than_auto_running(kappa):
    """Fail-closed by default: an unsure gate must not pick AUTO."""
    gate = ToolCallGate(manifold_dim=MANIFOLD_DIM, confidence_threshold=0.8)
    for d in gate.decide(kappa):
        if d.confidence < 0.8:
            assert d.route != AUTO


def test_low_confidence_auto_is_escalated():
    gate = ToolCallGate(manifold_dim=MANIFOLD_DIM, confidence_threshold=0.99)
    with torch.no_grad():  # force a confident-looking AUTO
        gate.route.proj.weight.zero_()
        gate.route.proj.bias.copy_(torch.tensor([10.0, 0.0, 0.0, 0.0]))
    d = gate.decide(torch.zeros(1, MANIFOLD_DIM))[0]
    assert d.route == AUTO and not d.escalated, "0.99 mass should stand"

    gate.confidence_threshold = 0.999999
    d = gate.decide(torch.zeros(1, MANIFOLD_DIM))[0]
    assert d.escalated and d.route == ESCALATION_ROUTE


def test_escalation_target_is_not_reject():
    """Refusing everything uncertain would make the gate unusable."""
    assert ESCALATION_ROUTE == CONFIRM


def test_high_intervention_routes_are_not_downgraded():
    """Uncertainty is never a reason to move a command toward running."""
    gate = ToolCallGate(manifold_dim=MANIFOLD_DIM, confidence_threshold=0.99)
    with torch.no_grad():
        gate.route.proj.weight.zero_()
        gate.route.proj.bias.copy_(torch.tensor([0.0, 0.0, 0.0, 0.1]))
    d = gate.decide(torch.zeros(1, MANIFOLD_DIM))[0]
    assert d.route == REJECT and not d.escalated


def test_decision_repr_and_helpers():
    d = GateDecision(route=AUTO, confidence=0.9, safety_risk=0.1)
    assert d.name == "auto" and d.may_auto_run
    assert not GateDecision(route=REJECT, confidence=0.9,
                            safety_risk=0.9).may_auto_run
    assert "auto" in repr(d)
    assert len(ROUTE_NAMES) == ToolCallGate.N_ROUTES


def test_decide_does_not_build_a_graph(kappa):
    gate = ToolCallGate(manifold_dim=MANIFOLD_DIM)
    k = kappa.clone().requires_grad_(True)
    gate.decide(k)
    assert k.grad is None


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------

def test_gate_loss_is_differentiable(kappa):
    gate = ToolCallGate(manifold_dim=MANIFOLD_DIM)
    loss = gate.loss(kappa, torch.randint(0, 4, (16,)), torch.ones(16))
    assert torch.isfinite(loss)
    loss.backward()
    assert gate.route.proj.weight.grad is not None


def test_gate_learns_a_separable_routing_signal():
    torch.manual_seed(0)
    gate = ToolCallGate(manifold_dim=MANIFOLD_DIM)
    k = torch.randn(64, MANIFOLD_DIM)
    # A route that is a clean function of two dimensions.
    target = ((k[:, 0] > 0).long() * 2 + (k[:, 1] > 0).long())
    opt = torch.optim.Adam(gate.parameters(), lr=0.05)
    first = gate.loss(k, target).item()
    for _ in range(300):
        opt.zero_grad()
        loss = gate.loss(k, target)
        loss.backward()
        opt.step()
    assert loss.item() < first * 0.5
    acc = (gate.route(k) == target).float().mean().item()
    assert acc > 0.9, f"gate only reached {acc:.2f} on a separable signal"


def test_gate_shares_the_safety_dimension():
    """The gate must reuse kappa[10], not learn a rival notion of danger."""
    from bash_mantis.models.typed_heads import KAPPA_SAFETY_MODE
    assert ToolCallGate(manifold_dim=MANIFOLD_DIM).safety.dim == KAPPA_SAFETY_MODE


# --------------------------------------------------------------------------
# Risk report
# --------------------------------------------------------------------------

def test_risk_report_counts_unsafe_auto():
    decisions = [
        GateDecision(AUTO, 0.9, 0.1),     # truth CONFIRM -> unsafe
        GateDecision(CONFIRM, 0.9, 0.9),  # truth CONFIRM -> correct
        GateDecision(SANDBOX, 0.9, 0.1),  # truth AUTO    -> overcautious
    ]
    r = gate_risk_report(decisions, [CONFIRM, CONFIRM, AUTO])
    assert r["n"] == 3
    assert r["unsafe_auto"] == pytest.approx(1 / 3)
    assert r["overcautious"] == pytest.approx(1 / 3)
    assert r["accuracy"] == pytest.approx(1 / 3)


def test_risk_report_skips_unlabelled():
    r = gate_risk_report([GateDecision(AUTO, 0.9, 0.1)] * 2, [None, AUTO])
    assert r["n"] == 1 and r["accuracy"] == 1.0


def test_risk_report_handles_nothing_labelled():
    assert gate_risk_report([GateDecision(AUTO, 0.9, 0.1)], [None]) == {"n": 0}


# --------------------------------------------------------------------------
# Integration
# --------------------------------------------------------------------------

def test_model_exposes_the_gate():
    from bash_mantis.models.bash_mantis_model import BashMantisModel
    m = BashMantisModel()
    out = m(torch.randint(0, 320, (3, 64)))
    decisions = m.tool_gate.decide(out.kappa)
    assert len(decisions) == 3
    assert all(0 <= d.route < ToolCallGate.N_ROUTES for d in decisions)


def test_gate_is_cheap():
    """It must stay a rounding error on a 672K model."""
    gate = ToolCallGate(manifold_dim=MANIFOLD_DIM)
    assert sum(p.numel() for p in gate.parameters()) < 100
