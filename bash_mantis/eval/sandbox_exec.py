"""Sandbox executor for Bash-MANTIS evaluation.

Runs Bash scripts in an isolated sandbox environment with:
- Temporary isolated directory
- No network
- No sudo
- Whitelisted binaries only
- Resource limits

Failures are classified rather than treated as a single undifferentiated
"wrong" signal.  A quoting slip and a fundamentally wrong algorithm are
both non-matching executions, but only the former is worth retrying, and
an environment failure is not the model's fault at all.  Training and
search both need that distinction.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path


class FailClass:
    """Failure taxonomy for a sandboxed Bash execution.

    Ordered roughly by how much blame attaches to the model:

      OK        - executed cleanly, exit 0
      SYNTAX    - shell could not parse the script (repairable)
      RUNTIME   - parsed and ran, but errored (repairable)
      SEMANTIC  - ran cleanly, but observable behavior differs from target
                  (right shape, wrong detail — "weak but underexplored")
      ENV       - missing binary / permission denied; the sandbox is at
                  fault, not the script (should not penalize the model)
      TIMEOUT   - hard-unrecoverable
    """

    OK = "ok"
    SYNTAX = "syntax"
    RUNTIME = "runtime"
    SEMANTIC = "semantic"
    ENV = "env"
    TIMEOUT = "timeout"

    #: Classes worth another attempt.
    REPAIRABLE = frozenset({SYNTAX, RUNTIME, SEMANTIC})
    #: Classes that say nothing about script quality.
    BLAMELESS = frozenset({ENV})

    # Relative training weight per class. ENV is zeroed so a broken
    # sandbox never becomes a gradient signal; SYNTAX is cheap because
    # it is a mechanical slip; SEMANTIC is the expensive one because the
    # script ran fine and still did the wrong thing.
    WEIGHTS = {
        OK: 0.0,
        SYNTAX: 0.4,
        RUNTIME: 0.7,
        SEMANTIC: 1.0,
        ENV: 0.0,
        TIMEOUT: 1.0,
    }


# stderr fragments that indicate the sandbox, not the script, is at fault.
_ENV_MARKERS = (
    "command not found",
    "permission denied",
    "cannot execute binary file",
    "no such file or directory: /usr",
)

_SYNTAX_MARKERS = (
    "syntax error",
    "unexpected end of file",
    "unexpected token",
    "unterminated quoted string",
    "missing `]'",
)


@dataclass
class Observables:
    """Observable behavior of a Bash script execution."""
    exit_code: int
    stdout: str
    stderr: str
    fs_delta: dict[str, str]  # filename -> content hash for created/modified files
    timed_out: bool = False
    fail_class: str = FailClass.OK

    def matches(self, other: "Observables") -> bool:
        """Check if two observables are equivalent."""
        return (
            self.exit_code == other.exit_code
            and self.stdout.strip() == other.stdout.strip()
            and self.fs_delta == other.fs_delta
        )

    @property
    def recoverable(self) -> bool:
        """True if another attempt at this script is worth the compute."""
        return self.fail_class in FailClass.REPAIRABLE

    @property
    def blameless(self) -> bool:
        """True if the failure is the sandbox's fault, not the script's."""
        return self.fail_class in FailClass.BLAMELESS

    @property
    def loss_weight(self) -> float:
        """Relative training weight for this outcome."""
        return FailClass.WEIGHTS.get(self.fail_class, 1.0)

    def to_dict(self) -> dict:
        return {
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "fs_delta": self.fs_delta,
            "timed_out": self.timed_out,
            "fail_class": self.fail_class,
        }


def classify_failure(exit_code: int, stderr: str, timed_out: bool = False) -> str:
    """Classify a completed execution into a FailClass.

    Only inspects the execution itself; SEMANTIC can't be detected here
    because it requires a target to compare against (see
    ``SandboxExecutor.execute_and_compare``).
    """
    if timed_out:
        return FailClass.TIMEOUT

    low = stderr.lower()

    # Environment problems first — they masquerade as script errors but
    # carry no information about script quality.
    if exit_code in (126, 127) or any(m in low for m in _ENV_MARKERS):
        return FailClass.ENV

    if exit_code == 0:
        return FailClass.OK

    # bash exits 2 on parse failure, but also check stderr since the exit
    # code alone is ambiguous with builtin misuse.
    if exit_code == 2 or any(m in low for m in _SYNTAX_MARKERS):
        return FailClass.SYNTAX

    return FailClass.RUNTIME


class SandboxExecutor:
    """Execute Bash scripts in a restricted sandbox."""

    def __init__(
        self,
        timeout_seconds: int = 5,
        max_memory_mb: int = 64,
        allowed_binaries: list[str] | None = None,
        base_temp_dir: str = "/tmp/mantis_sandbox",
        max_workers: int = 16,
    ):
        self.timeout = timeout_seconds
        self.max_memory_mb = max_memory_mb
        self.base_temp_dir = base_temp_dir
        self.max_workers = max_workers
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
        base = self.base_temp_dir
        if base and not os.path.exists(base):
            os.makedirs(base, exist_ok=True)
        sandbox_dir = tempfile.mkdtemp(dir=base, prefix="mantis_")

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
                    fail_class=classify_failure(result.returncode, result.stderr),
                )
            except subprocess.TimeoutExpired:
                return Observables(
                    exit_code=-1,
                    stdout="",
                    stderr="timeout",
                    fs_delta={},
                    timed_out=True,
                    fail_class=FailClass.TIMEOUT,
                )
        finally:
            shutil.rmtree(sandbox_dir, ignore_errors=True)

    def execute_batch(
        self,
        scripts: list[str],
        setup_files: dict[str, str] | None = None,
    ) -> list[Observables]:
        """Execute many scripts concurrently.

        Each run is an independent subprocess in its own temp dir, so
        these parallelize cleanly — ``subprocess.run`` releases the GIL
        while waiting. Wall-clock cost drops from N*timeout to ~timeout.
        """
        if not scripts:
            return []
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            return list(pool.map(lambda s: self.execute(s, setup_files), scripts))

    def execute_and_compare(
        self,
        script_pred: str,
        script_target: str,
        setup_files: dict[str, str] | None = None,
    ) -> tuple[Observables, Observables, bool]:
        """Execute two scripts and compare their observables.

        Upgrades the prediction's fail_class to SEMANTIC when it ran
        cleanly but diverged from the target — a clean run that produces
        the wrong answer is a different (and worse) failure than one that
        crashed.
        """
        obs_pred, obs_target = self.execute_batch(
            [script_pred, script_target], setup_files
        )
        match = obs_pred.matches(obs_target)

        if not match and obs_pred.fail_class == FailClass.OK:
            obs_pred.fail_class = FailClass.SEMANTIC

        return obs_pred, obs_target, match

    def check_equivalence(
        self,
        scripts: list[str],
        setup_files: dict[str, str] | None = None,
    ) -> list[list[bool]]:
        """Check pairwise equivalence of multiple scripts."""
        obs_list = self.execute_batch(scripts, setup_files)
        n = len(scripts)
        equiv = [[False] * n for _ in range(n)]
        for i in range(n):
            equiv[i][i] = True
            for j in range(i + 1, n):
                match = obs_list[i].matches(obs_list[j])
                equiv[i][j] = match
                equiv[j][i] = match
        return equiv
