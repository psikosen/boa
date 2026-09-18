#!/usr/bin/env python3
"""Benchmark the evaluation pipeline and the typed manifold heads.

Two things worth measuring after the failure-taxonomy / parallel-sandbox /
typed-head changes:

  1. Sandbox throughput. Execution dominates eval wall-clock at a 5s
     timeout, and the serial-to-parallel change is the single largest
     speedup available. This reports the real multiplier on this host.

  2. Typed-head calibration. The heads are only useful if their stated
     confidence matches observed accuracy -- that is the entire claim
     that makes `if confidence < tau: sample more` sound rather than
     superstitious. An untrained model should show poor ECE here; the
     number is a baseline to beat after training with lambda_calib > 0.

Usage:
    python scripts/benchmark_eval.py              # everything
    python scripts/benchmark_eval.py --sandbox    # sandbox only (no torch)
    python scripts/benchmark_eval.py --n 32
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bash_mantis.eval.sandbox_exec import FailClass, SandboxExecutor  # noqa: E402
from bash_mantis.eval.syntax_check import SyntaxChecker  # noqa: E402


# A spread of outcomes so the taxonomy is exercised, not just the happy path.
PROBE_SCRIPTS = [
    "echo hello",                      # ok
    "echo a; exit 1",                  # runtime
    "if then",                         # syntax
    "nosuchcmd_xyz",                   # env
    "for i in 1 2 3; do echo $i; done",# ok
    "cat missing_file_xyz",            # runtime
]


def _fmt(seconds: float) -> str:
    return f"{seconds * 1000:.0f}ms" if seconds < 1 else f"{seconds:.2f}s"


def bench_sandbox(n: int, repeats: int) -> None:
    print("=" * 66)
    print("SANDBOX THROUGHPUT")
    print("=" * 66)

    sb = SandboxExecutor(timeout_seconds=5, base_temp_dir="/tmp/mantis_bench")
    scripts = [PROBE_SCRIPTS[i % len(PROBE_SCRIPTS)] for i in range(n)]

    serial, parallel = [], []
    for _ in range(repeats):
        t0 = time.perf_counter()
        [sb.execute(s) for s in scripts]
        serial.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        sb.execute_batch(scripts)
        parallel.append(time.perf_counter() - t0)

    s, p = statistics.median(serial), statistics.median(parallel)
    print(f"  scripts           {n}   (workers={sb.max_workers}, repeats={repeats})")
    print(f"  serial            {_fmt(s)}")
    print(f"  parallel          {_fmt(p)}")
    print(f"  speedup           {s / max(p, 1e-9):.1f}x")
    print(f"  per-script serial {_fmt(s / n)}  ->  parallel {_fmt(p / n)}")

    obs = sb.execute_batch(scripts)
    counts: dict[str, int] = {}
    for o in obs:
        counts[o.fail_class] = counts.get(o.fail_class, 0) + 1
    print(f"\n  fail_class spread {counts}")
    blameless = sum(1 for o in obs if o.blameless)
    print(f"  blameless (zero-weight) {blameless}/{n}"
          f"  -- excluded from the gradient")

    # A slow script makes the parallel win unambiguous.
    slow = ["sleep 0.3; echo ok"] * min(n, 16)
    t0 = time.perf_counter(); [sb.execute(x) for x in slow]; ss = time.perf_counter() - t0
    t0 = time.perf_counter(); sb.execute_batch(slow);        sp = time.perf_counter() - t0
    print(f"\n  {len(slow)}x 'sleep 0.3': {_fmt(ss)} -> {_fmt(sp)}"
          f"  ({ss / max(sp, 1e-9):.1f}x)")

    sc = SyntaxChecker()
    many = ["echo ok; if true; then echo y; fi"] * (n * 4)
    t0 = time.perf_counter(); [sc.check(x) for x in many]; cs = time.perf_counter() - t0
    t0 = time.perf_counter(); sc.check_batch(many);        cp = time.perf_counter() - t0
    print(f"  {len(many)}x 'bash -n':   {_fmt(cs)} -> {_fmt(cp)}"
          f"  ({cs / max(cp, 1e-9):.1f}x)")


def bench_typed_heads(n: int) -> None:
    try:
        import torch
    except ImportError:
        print("\n(skipping typed-head benchmark: torch not installed)")
        return

    from bash_mantis.models.typed_heads import (
        TypedManifoldHeads, expected_calibration_error,
    )

    print("\n" + "=" * 66)
    print("TYPED HEADS")
    print("=" * 66)

    torch.manual_seed(0)
    heads = TypedManifoldHeads()
    kappa = torch.randn(n, 12)

    t0 = time.perf_counter()
    for _ in range(100):
        out = heads(kappa)
    elapsed = (time.perf_counter() - t0) / 100
    print(f"  forward ({n} samples)  {_fmt(elapsed)}")
    print(f"  params                 {sum(p.numel() for p in heads.parameters())}")

    cs = out["completion_score"]
    print(f"\n  completion_score  min={cs.min():.3f} max={cs.max():.3f} "
          f"mean={cs.mean():.3f}   (valid range [0, 2])")
    frac = (cs - cs.round()).abs().mean()
    print(f"  mean distance from an integer level: {frac:.3f}"
          f"   -- a Score is a position, not an index")

    # Untrained calibration: the baseline that training should improve on.
    print("\n  UNTRAINED calibration (baseline to beat):")
    safety_truth = (kappa[:, 10] > 0).float()
    ece_before = expected_calibration_error(out["safety_risk"], safety_truth)
    print(f"    safety ECE  {ece_before:.4f}")

    opt = torch.optim.Adam(heads.parameters(), lr=0.05)
    for _ in range(300):
        opt.zero_grad()
        heads.calibration_loss(kappa, safety_target=safety_truth).backward()
        opt.step()

    ece_after = expected_calibration_error(heads(kappa)["safety_risk"], safety_truth)
    print(f"  AFTER fitting a signal carried by kappa[10]:")
    print(f"    safety ECE  {ece_after:.4f}"
          f"   ({'improved' if ece_after < ece_before else 'NO IMPROVEMENT'})")
    if ece_after >= ece_before:
        print("    ^ investigate: the head should be able to fit its own dimension")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=24, help="scripts per benchmark")
    ap.add_argument("--repeats", type=int, default=3, help="timing repeats (median)")
    ap.add_argument("--sandbox", action="store_true", help="sandbox only, skip torch")
    ap.add_argument("--heads", action="store_true", help="typed heads only")
    args = ap.parse_args()

    if not args.heads:
        bench_sandbox(args.n, args.repeats)
    if not args.sandbox:
        bench_typed_heads(args.n)

    print("\n" + "=" * 66)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
