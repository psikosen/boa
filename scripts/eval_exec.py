#!/usr/bin/env python3
"""Evaluate Bash-MANTIS execution correctness in sandbox.

Usage:
    python scripts/eval_exec.py --checkpoint outputs/stage0/final_model.pt
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from bash_mantis.utils.config import load_config
from bash_mantis.tokenization.byte_tokenizer import ByteTokenizer
from bash_mantis.models.bash_mantis_model import BashMantisModel
from bash_mantis.eval.sandbox_exec import SandboxExecutor
from bash_mantis.eval.metrics import BashMetrics, EvalResults


# Test cases: (intent, expected_command, setup_files)
EXEC_TESTS = [
    {
        "intent": "list all txt files",
        "target": "find . -type f -name '*.txt'",
        "setup": {"a.txt": "hello", "b.txt": "world", "c.log": "skip"},
    },
    {
        "intent": "count lines in data.txt",
        "target": "wc -l data.txt",
        "setup": {"data.txt": "line1\nline2\nline3\n"},
    },
    {
        "intent": "create a backup directory",
        "target": "mkdir -p backup",
        "setup": {},
    },
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    tokenizer = ByteTokenizer(config.model.vocab_size)
    model = BashMantisModel.from_config(config)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    sandbox = SandboxExecutor(timeout_seconds=5)

    print("Bash-MANTIS Execution Evaluation")
    print("=" * 50)

    for test in EXEC_TESTS:
        print(f"\nIntent: {test['intent']}")
        print(f"Target: {test['target']}")

        # Execute target
        obs_target = sandbox.execute(test["target"], test.get("setup"))
        print(f"  Target exit code: {obs_target.exit_code}")
        print(f"  Target stdout: {obs_target.stdout[:80]}")

        # In a real eval, we would generate a prediction and compare
        # For now, just show the target execution results
        print(f"  FS delta: {obs_target.fs_delta}")

    print("\nExecution evaluation framework ready.")
    print("Full evaluation requires trained model generating predictions.")


if __name__ == "__main__":
    main()
