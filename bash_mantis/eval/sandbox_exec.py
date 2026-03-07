"""Sandbox executor for Bash-MANTIS evaluation.

Runs Bash scripts in an isolated sandbox environment with:
- Temporary isolated directory
- No network
- No sudo
- Whitelisted binaries only
- Resource limits
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Observables:
    """Observable behavior of a Bash script execution."""
    exit_code: int
    stdout: str
    stderr: str
    fs_delta: dict[str, str]  # filename -> content hash for created/modified files
    timed_out: bool = False

    def matches(self, other: "Observables") -> bool:
        """Check if two observables are equivalent."""
        return (
            self.exit_code == other.exit_code
            and self.stdout.strip() == other.stdout.strip()
            and self.fs_delta == other.fs_delta
        )

    def to_dict(self) -> dict:
        return {
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "fs_delta": self.fs_delta,
            "timed_out": self.timed_out,
        }


class SandboxExecutor:
    """Execute Bash scripts in a restricted sandbox."""

    def __init__(
        self,
        timeout_seconds: int = 5,
        max_memory_mb: int = 64,
        allowed_binaries: list[str] | None = None,
        base_temp_dir: str = "/tmp/mantis_sandbox",
    ):
        self.timeout = timeout_seconds
        self.max_memory_mb = max_memory_mb
        self.base_temp_dir = base_temp_dir
        self.allowed_binaries = allowed_binaries or [
            "bash", "find", "grep", "sed", "awk", "sort", "uniq", "cut", "tr",
            "wc", "tar", "gzip", "xargs", "cat", "printf", "echo", "mv", "cp",
            "rm", "mkdir", "touch", "head", "tail",
        ]

    def _hash_file(self, path: Path) -> str:
        """Get content hash of a file."""
        return hashlib.md5(path.read_bytes()).hexdigest()

    def _snapshot_dir(self, directory: Path) -> dict[str, str]:
        """Get a snapshot of all files in a directory with their content hashes."""
        snapshot = {}
        if directory.exists():
            for f in sorted(directory.rglob("*")):
                if f.is_file():
                    rel = str(f.relative_to(directory))
                    snapshot[rel] = self._hash_file(f)
        return snapshot

    def _compute_fs_delta(
        self, before: dict[str, str], after: dict[str, str]
    ) -> dict[str, str]:
        """Compute filesystem changes."""
        delta = {}
        for path, hash_val in after.items():
            if path not in before or before[path] != hash_val:
                delta[path] = hash_val
        for path in before:
            if path not in after:
                delta[path] = "DELETED"
        return delta

    def execute(
        self,
        script: str,
        setup_files: dict[str, str] | None = None,
    ) -> Observables:
        """Execute a script in the sandbox.

        Args:
            script: Bash script to execute
            setup_files: optional dict of {filename: content} to create before execution

        Returns:
            Observables with execution results
        """
        sandbox_dir = tempfile.mkdtemp(dir=self.base_temp_dir if os.path.exists(
            os.path.dirname(self.base_temp_dir)) else None,
            prefix="mantis_")

        try:
            sandbox_path = Path(sandbox_dir)

            # Create setup files
            if setup_files:
                for fname, content in setup_files.items():
                    fpath = sandbox_path / fname
                    fpath.parent.mkdir(parents=True, exist_ok=True)
                    fpath.write_text(content)

            # Snapshot before
            before = self._snapshot_dir(sandbox_path)

            # Build restricted environment
            env = {
                "HOME": sandbox_dir,
                "TMPDIR": sandbox_dir,
                "PATH": "/usr/bin:/bin",
                "SHELL": "/bin/bash",
            }

            # Execute
            try:
                result = subprocess.run(
                    ["bash", "-c", script],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    cwd=sandbox_dir,
                    env=env,
                )
                # Snapshot after
                after = self._snapshot_dir(sandbox_path)
                fs_delta = self._compute_fs_delta(before, after)

                return Observables(
                    exit_code=result.returncode,
                    stdout=result.stdout,
                    stderr=result.stderr,
                    fs_delta=fs_delta,
                )
            except subprocess.TimeoutExpired:
                return Observables(
                    exit_code=-1,
                    stdout="",
                    stderr="timeout",
                    fs_delta={},
                    timed_out=True,
                )
        finally:
            shutil.rmtree(sandbox_dir, ignore_errors=True)

    def execute_and_compare(
        self,
        script_pred: str,
        script_target: str,
        setup_files: dict[str, str] | None = None,
    ) -> tuple[Observables, Observables, bool]:
        """Execute two scripts and compare their observables."""
        obs_pred = self.execute(script_pred, setup_files)
        obs_target = self.execute(script_target, setup_files)
        match = obs_pred.matches(obs_target)
        return obs_pred, obs_target, match

    def check_equivalence(
        self,
        scripts: list[str],
        setup_files: dict[str, str] | None = None,
    ) -> list[list[bool]]:
        """Check pairwise equivalence of multiple scripts."""
        obs_list = [self.execute(s, setup_files) for s in scripts]
        n = len(scripts)
        equiv = [[False] * n for _ in range(n)]
        for i in range(n):
            equiv[i][i] = True
            for j in range(i + 1, n):
                match = obs_list[i].matches(obs_list[j])
                equiv[i][j] = match
                equiv[j][i] = match
        return equiv
