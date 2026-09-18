"""Derive typed-head supervision labels from execution outcomes.

The typed manifold heads are only meaningful if their targets come from
what actually happened rather than from what a human thought looked
good. This module is the bridge: sandbox Observables and raw script text
in, tensors the heads can be trained against out.

Three labels:

  completion  Score, 3 ordered levels:
                0  did not parse / did not run
                1  ran, but observable behavior differs from target
                2  ran and matched
              The ordering is real — level 1 is strictly closer to
              working than level 0 — which is what lets the ScoreHead's
              ordinal link do useful work.

  repair      Noul: did a fix_bash attempt actually repair the script?
              Only defined when the input was broken to begin with.

  safety      Noul: does this script contain a destructive operation?
              Derived from the text, not from execution — we emphatically
              do not want to run `rm -rf /` to find out.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from bash_mantis.eval.sandbox_exec import FailClass, Observables

if TYPE_CHECKING:  # pragma: no cover
    import torch

# torch is imported lazily inside build_labels so the rest of this module
# — and the eval package as a whole — stays usable in a sandbox-running
# harness that has no need for the training stack.


# Completion levels.
COMPLETION_FAILED = 0
COMPLETION_RAN_MISMATCH = 1
COMPLETION_MATCHED = 2


#: Patterns that make a script destructive. Kept deliberately broad —
#: a false positive costs a little precision on the safety head, while a
#: false negative teaches the model that something dangerous is fine.
_DESTRUCTIVE_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\brm\s+(-[a-zA-Z]*\s+)*-[a-zA-Z]*[rf]", "recursive/forced delete"),
    (r"\brm\s+.*\$\{?\w+", "delete with unquoted variable path"),
    (r"\bdd\s+.*\bof=", "raw device write"),
    (r"\bmkfs(\.\w+)?\b", "filesystem format"),
    (r">\s*/dev/(sd|nvme|hd)", "write to block device"),
    (r"\bchmod\s+(-[a-zA-Z]+\s+)*777\b", "world-writable permissions"),
    (r"\bshred\b", "secure erase"),
    (r"\btruncate\s+.*-s\s*0", "truncate to zero"),
    (r"-delete\b", "find -delete"),
    (r"\bcurl\b[^|]*\|\s*(sudo\s+)?(ba)?sh", "pipe remote script to shell"),
    (r"\bwget\b[^|]*\|\s*(sudo\s+)?(ba)?sh", "pipe remote script to shell"),
    (r"\b(shutdown|reboot|halt|poweroff)\b", "host power control"),
    (r":\(\)\s*\{\s*:\|:&\s*\}\s*;\s*:", "fork bomb"),
)

_COMPILED = tuple((re.compile(p), why) for p, why in _DESTRUCTIVE_PATTERNS)


def safety_reasons(script: str) -> list[str]:
    """Return human-readable reasons a script is considered destructive.

    Empty list means no destructive pattern matched.
    """
    return [why for rx, why in _COMPILED if rx.search(script)]


def is_destructive(script: str) -> bool:
    """True if the script contains any destructive operation."""
    return bool(safety_reasons(script))


def completion_level(obs: Observables, matched: bool) -> int:
    """Map an execution outcome onto the ordered completion scale.

    ENV failures are the awkward case: the script may have been fine and
    the sandbox simply lacked a binary. We score those as
    COMPLETION_FAILED here but callers should mask them out of the loss
    entirely via ``build_labels(..., drop_blameless=True)``.
    """
    if matched:
        return COMPLETION_MATCHED
    if obs.fail_class in (FailClass.OK, FailClass.SEMANTIC):
        # It ran to completion and simply did the wrong thing.
        return COMPLETION_RAN_MISMATCH
    return COMPLETION_FAILED


def build_labels(
    scripts: list[str],
    observables: list[Observables],
    matches: list[bool],
    was_broken: list[bool] | None = None,
    drop_blameless: bool = True,
    device: "torch.device | None" = None,
) -> "dict[str, torch.Tensor]":
    """Assemble typed-head targets for a batch.

    Args:
        scripts: generated Bash, one per sample
        observables: sandbox result per sample
        matches: whether each sample matched its target
        was_broken: for repair tasks, whether the input script was broken.
            Samples where this is False contribute no repair supervision
            (there was nothing to fix, so "did it fix it" is undefined).
        drop_blameless: exclude ENV failures from completion supervision,
            so a missing binary in the sandbox never becomes a gradient.
        device: target device for the returned tensors

    Returns:
        dict with ``completion_target`` (long), ``safety_target`` (float),
        ``repair_target`` (float), and the index tensors ``completion_idx``
        / ``repair_idx`` identifying which samples each target covers.
    """
    import torch

    n = len(scripts)
    if not (len(observables) == len(matches) == n):
        raise ValueError("scripts, observables and matches must be the same length")

    completion_vals: list[int] = []
    completion_idx: list[int] = []
    repair_vals: list[float] = []
    repair_idx: list[int] = []

    for i, (script, obs, matched) in enumerate(zip(scripts, observables, matches)):
        if not (drop_blameless and obs.blameless):
            completion_vals.append(completion_level(obs, matched))
            completion_idx.append(i)

        if was_broken is not None and was_broken[i]:
            repair_vals.append(1.0 if matched else 0.0)
            repair_idx.append(i)

    safety_vals = [1.0 if is_destructive(s) else 0.0 for s in scripts]

    def t(vals, dtype):
        return torch.tensor(vals, dtype=dtype, device=device)

    return {
        "completion_target": t(completion_vals, torch.long),
        "completion_idx": t(completion_idx, torch.long),
        "safety_target": t(safety_vals, torch.float),
        "repair_target": t(repair_vals, torch.float),
        "repair_idx": t(repair_idx, torch.long),
    }
