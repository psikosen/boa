"""Syntax checking for Bash-MANTIS outputs.

Uses `bash -n` to validate shell syntax without executing commands.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path


class SyntaxChecker:
    """Validates Bash syntax using `bash -n`."""

    def check(self, script: str) -> bool:
        """Check if a Bash script passes syntax validation.

        Returns True if `bash -n` succeeds, False otherwise.
        """
        if not script.strip():
            return False

        try:
            result = subprocess.run(
                ["bash", "-n"],
                input=script,
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    def check_batch(self, scripts: list[str]) -> list[bool]:
        """Check syntax for a batch of scripts."""
        return [self.check(s) for s in scripts]

    def get_errors(self, script: str) -> str:
        """Get syntax error messages from bash -n."""
        if not script.strip():
            return "empty script"

        try:
            result = subprocess.run(
                ["bash", "-n"],
                input=script,
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return ""
            return result.stderr.strip()
        except subprocess.TimeoutExpired:
            return "timeout"
        except FileNotFoundError:
            return "bash not found"

    def syntax_pass_rate(self, scripts: list[str]) -> float:
        """Compute the fraction of scripts that pass syntax checking."""
        if not scripts:
            return 0.0
        results = self.check_batch(scripts)
        return sum(results) / len(results)
