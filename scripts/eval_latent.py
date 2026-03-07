#!/usr/bin/env python3
"""Evaluate Bash-MANTIS latent space structure.

Checks:
  - Correct states are more stable than incorrect
  - Equivalent commands cluster
  - Text-Bash alignment in latent space

Usage:
    python scripts/eval_latent.py --checkpoint outputs/stage6/final_model.pt
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from bash_mantis.utils.config import load_config
from bash_mantis.tokenization.byte_tokenizer import ByteTokenizer
from bash_mantis.models.bash_mantis_model import BashMantisModel
from bash_mantis.eval.latent_analysis import LatentAnalyzer


# Equivalent command pairs for testing clustering
EQUIV_PAIRS = [
    ("grep -r foo .", "find . -type f -exec grep -H foo {} +"),
    ('find . -name "*.log" | xargs gzip', 'find . -name "*.log" -print0 | xargs -0 gzip'),
    ("cat file | sort | uniq", "sort file | uniq"),
]

# Non-equivalent pairs
NON_EQUIV_PAIRS = [
    ("ls -la", "rm -rf ."),
    ("grep foo bar.txt", "sed 's/foo/bar/g' input.txt"),
    ("wc -l *.txt", "tar czf archive.tar.gz *.txt"),
]


def encode_command(model, tokenizer, cmd, device):
    """Encode a command and get its kappa state."""
    prompt = f"[TASK] write_bash\n[INTENT] execute\n[OUTPUT]\n{cmd}"
    ids = tokenizer.encode(prompt)
    ids = tokenizer.pad(ids, 256)
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)

    with torch.no_grad():
        output = model(input_ids)
    return output.kappa[0]


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
    device = torch.device("cpu")

    analyzer = LatentAnalyzer()

    print("Bash-MANTIS Latent Space Analysis")
    print("=" * 50)

    # Encode equivalent pairs
    equiv_kappa_pairs = []
    print("\nEquivalent command pairs:")
    for cmd_a, cmd_b in EQUIV_PAIRS:
        ka = encode_command(model, tokenizer, cmd_a, device)
        kb = encode_command(model, tokenizer, cmd_b, device)
        dist = (ka - kb).pow(2).sum().sqrt().item()
        equiv_kappa_pairs.append((ka, kb))
        print(f"  '{cmd_a[:40]}' <-> '{cmd_b[:40]}': dist={dist:.4f}")

    # Encode non-equivalent pairs
    non_equiv_kappa_pairs = []
    print("\nNon-equivalent command pairs:")
    for cmd_a, cmd_b in NON_EQUIV_PAIRS:
        ka = encode_command(model, tokenizer, cmd_a, device)
        kb = encode_command(model, tokenizer, cmd_b, device)
        dist = (ka - kb).pow(2).sum().sqrt().item()
        non_equiv_kappa_pairs.append((ka, kb))
        print(f"  '{cmd_a[:40]}' <-> '{cmd_b[:40]}': dist={dist:.4f}")

    # Run analysis
    kappa_correct = torch.stack([p[0] for p in equiv_kappa_pairs])
    metrics = analyzer.analyze(
        kappa_correct=kappa_correct,
        kappa_equiv_pairs=equiv_kappa_pairs,
        kappa_non_equiv_pairs=non_equiv_kappa_pairs,
    )

    print(f"\nLatent Metrics:")
    for k, v in metrics.to_dict().items():
        print(f"  {k}: {v:.4f}")

    # Check separation
    if metrics.mean_non_equiv_distance > metrics.mean_equiv_distance:
        print("\n  PASS: Non-equivalent commands are farther apart than equivalent ones")
    else:
        print("\n  FAIL: Latent separation not yet achieved (needs training)")


if __name__ == "__main__":
    main()
