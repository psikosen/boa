"""Evaluation package for Bash-MANTIS.

Most of this package is deliberately torch-free: syntax checking, sandbox
execution, the failure taxonomy, and metrics all run on subprocesses and
plain Python, so an eval harness can use them without installing the
training stack.

``LatentAnalyzer`` is the exception — it needs torch and numpy — so it is
resolved lazily (PEP 562). Importing the package no longer drags torch in
for callers who only want to run scripts in a sandbox.
"""

from bash_mantis.eval.syntax_check import SyntaxChecker
from bash_mantis.eval.sandbox_exec import (
    FailClass,
    Observables,
    SandboxExecutor,
    classify_failure,
)
from bash_mantis.eval.metrics import BashMetrics

__all__ = [
    "SyntaxChecker",
    "SandboxExecutor",
    "Observables",
    "FailClass",
    "classify_failure",
    "BashMetrics",
    "LatentAnalyzer",
]


def __getattr__(name: str):
    """Resolve torch-dependent members on first access."""
    if name == "LatentAnalyzer":
        from bash_mantis.eval.latent_analysis import LatentAnalyzer
        return LatentAnalyzer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
