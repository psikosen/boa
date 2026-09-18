"""Tests for the calibrated typed manifold heads.

These cover the tensor paths that could not be executed when the heads
were written (no torch in that environment). The ordinal math was
verified against a pure-Python mirror at the time; this is the real
thing, plus the shape/dtype plumbing a hand-check cannot catch.

Run: pytest tests/test_typed_heads.py -v
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from bash_mantis.models.typed_heads import (  # noqa: E402
    KAPPA_COMPLETION_CONFIDENCE,
    KAPPA_REPAIR_CONFIDENCE,
    KAPPA_SAFETY_MODE,
    NoulHead,
    ScoreHead,
    TypedManifoldHeads,
    expected_calibration_error,
)

MANIFOLD_DIM = 12


@pytest.fixture
def kappa():
    torch.manual_seed(0)
    return torch.randn(16, MANIFOLD_DIM)


# --------------------------------------------------------------------------
# NoulHead
# --------------------------------------------------------------------------

def test_noul_returns_probabilities(kappa):
    head = NoulHead(KAPPA_SAFETY_MODE)
    p = head(kappa)
    assert p.shape == (16,)
    assert ((p >= 0) & (p <= 1)).all()


def test_noul_is_monotone_in_its_dimension(kappa):
    """The whole point: kappa[dim] must drive the answer."""
    head = NoulHead(KAPPA_SAFETY_MODE)
    with torch.no_grad():
        head.scale.fill_(1.0)
    lo = head(kappa)
    bumped = kappa.clone()
    bumped[:, KAPPA_SAFETY_MODE] += 3.0
    assert (head(bumped) >= lo - 1e-6).all()


def test_noul_ignores_other_dimensions(kappa):
    """A head reading dim d must not be movable by any other dim."""
    head = NoulHead(KAPPA_SAFETY_MODE)
    before = head(kappa)
    perturbed = kappa.clone()
    for d in range(MANIFOLD_DIM):
        if d != KAPPA_SAFETY_MODE:
            perturbed[:, d] += 10.0
    assert torch.allclose(head(perturbed), before)


def test_noul_loss_decreases_with_training(kappa):
    """Sanity: the head can actually fit a signal carried by its dim."""
    head = NoulHead(KAPPA_SAFETY_MODE)
    targets = (kappa[:, KAPPA_SAFETY_MODE] > 0).float()
    opt = torch.optim.Adam(head.parameters(), lr=0.1)
    first = head.loss(kappa, targets).item()
    for _ in range(200):
        opt.zero_grad()
        loss = head.loss(kappa, targets)
        loss.backward()
        opt.step()
    assert loss.item() < first * 0.5


# --------------------------------------------------------------------------
# ScoreHead
# --------------------------------------------------------------------------

@pytest.mark.parametrize("levels", [2, 3, 5, 10])
def test_score_probabilities_are_a_distribution(kappa, levels):
    head = ScoreHead(KAPPA_COMPLETION_CONFIDENCE, levels=levels)
    p = head.probabilities(kappa)
    assert p.shape == (16, levels)
    assert (p >= 0).all(), "cumulative link produced a negative probability"
    assert torch.allclose(p.sum(-1), torch.ones(16), atol=1e-5)


@pytest.mark.parametrize("levels", [2, 3, 5])
def test_score_is_fractional_position_not_index(kappa, levels):
    """A Score is a probability-weighted mean in [0, levels-1]."""
    head = ScoreHead(KAPPA_COMPLETION_CONFIDENCE, levels=levels)
    s = head(kappa)
    assert s.shape == (16,)
    assert (s >= 0).all() and (s <= levels - 1).all()
    # It must genuinely land between levels, not snap to integers.
    assert not torch.allclose(s, s.round()), "score collapsed to an index"


def test_score_matches_manual_expectation(kappa):
    head = ScoreHead(KAPPA_COMPLETION_CONFIDENCE, levels=3)
    p = head.probabilities(kappa)
    manual = p[:, 0] * 0 + p[:, 1] * 1 + p[:, 2] * 2
    assert torch.allclose(head(kappa), manual, atol=1e-6)


def test_score_thresholds_strictly_increase():
    head = ScoreHead(KAPPA_COMPLETION_CONFIDENCE, levels=6)
    with torch.no_grad():
        head.theta_deltas.fill_(-8.0)  # softplus -> ~0, the degenerate case
    th = head._thresholds()
    assert (th[1:] > th[:-1]).all(), "ordering must hold even at extremes"


def test_score_monotone_in_its_dimension(kappa):
    head = ScoreHead(KAPPA_COMPLETION_CONFIDENCE, levels=3)
    with torch.no_grad():
        head.scale.fill_(1.0)
    prev = None
    for bump in [-4.0, -2.0, 0.0, 2.0, 4.0]:
        k = kappa.clone()
        k[:, KAPPA_COMPLETION_CONFIDENCE] = bump
        s = head(k)
        if prev is not None:
            assert (s >= prev - 1e-6).all(), "score fell as the latent rose"
        prev = s


def test_score_confidence_is_max_level_mass(kappa):
    head = ScoreHead(KAPPA_COMPLETION_CONFIDENCE, levels=3)
    c = head.confidence(kappa)
    assert torch.allclose(c, head.probabilities(kappa).max(-1).values)
    assert (c >= 1.0 / 3 - 1e-6).all(), "max mass cannot be below uniform"


def test_score_rejects_too_few_levels():
    with pytest.raises(ValueError):
        ScoreHead(0, levels=1)


def test_score_loss_is_finite_and_differentiable(kappa):
    head = ScoreHead(KAPPA_COMPLETION_CONFIDENCE, levels=3)
    targets = torch.randint(0, 3, (16,))
    loss = head.loss(kappa, targets)
    assert torch.isfinite(loss)
    loss.backward()
    assert head.theta_0.grad is not None
    assert torch.isfinite(head.theta_0.grad).all()


# --------------------------------------------------------------------------
# TypedManifoldHeads
# --------------------------------------------------------------------------

def test_typed_heads_forward_keys_and_ranges(kappa):
    heads = TypedManifoldHeads()
    out = heads(kappa)
    assert set(out) == {
        "repair_confidence", "safety_risk",
        "completion_score", "completion_confidence",
    }
    for k in ("repair_confidence", "safety_risk", "completion_confidence"):
        assert ((out[k] >= 0) & (out[k] <= 1)).all(), f"{k} outside [0,1]"
    cs = out["completion_score"]
    assert (cs >= 0).all() and (cs <= TypedManifoldHeads.COMPLETION_LEVELS - 1).all()


def test_typed_heads_read_the_declared_dimensions():
    """Guards against someone renumbering kappa without updating the heads."""
    heads = TypedManifoldHeads()
    assert heads.repair.dim == KAPPA_REPAIR_CONFIDENCE == 7
    assert heads.safety.dim == KAPPA_SAFETY_MODE == 10
    assert heads.completion.dim == KAPPA_COMPLETION_CONFIDENCE == 11


def test_calibration_loss_accepts_partial_targets(kappa):
    """Not every batch carries every label; missing ones must be skipped."""
    heads = TypedManifoldHeads()
    only_safety = heads.calibration_loss(kappa, safety_target=torch.ones(16))
    assert torch.isfinite(only_safety) and only_safety > 0

    none_at_all = heads.calibration_loss(kappa)
    assert none_at_all.item() == 0.0


def test_calibration_loss_sums_heads(kappa):
    heads = TypedManifoldHeads()
    s = torch.ones(16)
    c = torch.randint(0, 3, (16,))
    total = heads.calibration_loss(kappa, safety_target=s, completion_target=c)
    parts = (
        heads.safety.loss(kappa, s) + heads.completion.loss(kappa, c)
    )
    assert torch.allclose(total, parts, atol=1e-6)


def test_calibration_loss_backprops_to_kappa():
    """Gradient must reach the manifold, not just the head parameters."""
    heads = TypedManifoldHeads()
    k = torch.randn(8, MANIFOLD_DIM, requires_grad=True)
    heads.calibration_loss(
        k,
        safety_target=torch.ones(8),
        completion_target=torch.randint(0, 3, (8,)),
    ).backward()
    assert k.grad is not None
    touched = k.grad.abs().sum(0) > 0
    assert touched[KAPPA_SAFETY_MODE], "safety head did not reach kappa[10]"
    assert touched[KAPPA_COMPLETION_CONFIDENCE], "completion head did not reach kappa[11]"
    # And it must NOT smear gradient into dimensions it does not read.
    assert not touched[0] and not touched[5]


# --------------------------------------------------------------------------
# Calibration measurement
# --------------------------------------------------------------------------

def test_ece_zero_for_perfectly_calibrated():
    torch.manual_seed(1)
    p = torch.full((4000,), 0.5)
    o = (torch.rand(4000) < 0.5).float()
    assert expected_calibration_error(p, o) < 0.05


def test_ece_detects_miscalibration():
    torch.manual_seed(2)
    p = torch.rand(4000)
    good = (torch.rand(4000) < p).float()
    assert expected_calibration_error(p, good) < expected_calibration_error(p, 1 - good)


def test_ece_handles_empty():
    assert expected_calibration_error(torch.empty(0), torch.empty(0)) == 0.0
