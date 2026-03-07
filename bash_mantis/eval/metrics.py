"""Evaluation metrics for Bash-MANTIS.

Four evaluation axes:
  1. Syntax quality (bash -n pass rate)
  2. Execution correctness (sandbox behavioral match)
  3. Instruction following (command match, top-k success)
  4. Function quality (validity, unit tests, safety patterns)
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from bash_mantis.eval.syntax_check import SyntaxChecker
from bash_mantis.eval.sandbox_exec import SandboxExecutor


@dataclass
class EvalResults:
    """Container for evaluation results."""
    syntax_pass_rate: float = 0.0
    exec_match_rate: float = 0.0
    exact_match_rate: float = 0.0
    normalized_match_rate: float = 0.0
    n_samples: int = 0

    def to_dict(self) -> dict:
        return {
            "syntax_pass_rate": self.syntax_pass_rate,
            "exec_match_rate": self.exec_match_rate,
            "exact_match_rate": self.exact_match_rate,
            "normalized_match_rate": self.normalized_match_rate,
            "n_samples": self.n_samples,
        }


class BashMetrics:
    """Compute evaluation metrics for Bash generation."""

    def __init__(
        self,
        syntax_checker: SyntaxChecker | None = None,
        sandbox: SandboxExecutor | None = None,
    ):
        self.syntax = syntax_checker or SyntaxChecker()
        self.sandbox = sandbox

    @staticmethod
    def normalize_bash(cmd: str) -> str:
        """Normalize a Bash command for fuzzy matching.

        Strips whitespace, normalizes quoting, sorts flags.
        """
        cmd = cmd.strip()
        # Collapse whitespace
        cmd = re.sub(r"\s+", " ", cmd)
        # Remove trailing semicolons
        cmd = cmd.rstrip(";").strip()
        return cmd

    def exact_match(self, pred: str, target: str) -> bool:
        """Check exact string match."""
        return pred.strip() == target.strip()

    def normalized_match(self, pred: str, target: str) -> bool:
        """Check normalized match."""
        return self.normalize_bash(pred) == self.normalize_bash(target)

    def evaluate(
        self,
        predictions: list[str],
        targets: list[str],
        setup_files: dict[str, str] | None = None,
    ) -> EvalResults:
        """Run full evaluation on a list of prediction/target pairs."""
        n = len(predictions)
        if n == 0:
            return EvalResults()

        # Syntax
        syntax_results = self.syntax.check_batch(predictions)
        syntax_pass_rate = sum(syntax_results) / n

        # Exact and normalized match
        exact_matches = sum(
            self.exact_match(p, t) for p, t in zip(predictions, targets)
        )
        norm_matches = sum(
            self.normalized_match(p, t) for p, t in zip(predictions, targets)
        )

        # Execution match
        exec_matches = 0
        if self.sandbox:
            for pred, target in zip(predictions, targets):
                try:
                    _, _, match = self.sandbox.execute_and_compare(
                        pred, target, setup_files
                    )
                    if match:
                        exec_matches += 1
                except Exception:
                    pass

        return EvalResults(
            syntax_pass_rate=syntax_pass_rate,
            exec_match_rate=exec_matches / n if self.sandbox else 0.0,
            exact_match_rate=exact_matches / n,
            normalized_match_rate=norm_matches / n,
            n_samples=n,
        )
