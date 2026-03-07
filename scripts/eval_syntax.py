#!/usr/bin/env python3
"""Evaluate Bash-MANTIS syntax generation quality.

Usage:
    python scripts/eval_syntax.py --checkpoint outputs/stage0/final_model.pt
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from bash_mantis.utils.config import load_config, MantisConfig
from bash_mantis.tokenization.byte_tokenizer import ByteTokenizer
from bash_mantis.models.bash_mantis_model import BashMantisModel
from bash_mantis.eval.syntax_check import SyntaxChecker
from bash_mantis.eval.metrics import BashMetrics


TEST_PROMPTS = [
    "[TASK] write_bash\n[INTENT] list all files in the current directory\n[OUTPUT]",
    "[TASK] write_bash\n[INTENT] find all log files larger than 10MB\n[OUTPUT]",
    "[TASK] write_bash\n[INTENT] count lines in all csv files\n[OUTPUT]",
    "[TASK] write_bash\n[INTENT] compress all txt files with gzip\n[OUTPUT]",
    "[TASK] write_bash\n[INTENT] find files modified in the last 7 days\n[OUTPUT]",
]


def generate(model, tokenizer, prompt, max_new_tokens=64, temperature=0.8):
    """Simple greedy/sampling generation."""
    model.eval()
    device = next(model.parameters()).device
    input_ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
    input_ids = torch.tensor([input_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            if input_ids.shape[1] >= 256:
                break
            output = model(input_ids)
            next_logits = output.logits[:, -1, :] / temperature
            probs = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            if next_token.item() == tokenizer.EOS:
                break
            input_ids = torch.cat([input_ids, next_token], dim=1)

    return tokenizer.decode(input_ids[0].tolist())


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

    checker = SyntaxChecker()
    print("Bash-MANTIS Syntax Evaluation")
    print("=" * 50)

    generated_scripts = []
    for prompt in TEST_PROMPTS:
        output = generate(model, tokenizer, prompt)
        # Extract the part after [OUTPUT]
        if "[OUTPUT]" in output:
            script = output.split("[OUTPUT]")[-1].strip()
        else:
            script = output.strip()
        generated_scripts.append(script)
        valid = checker.check(script)
        status = "PASS" if valid else "FAIL"
        print(f"\n[{status}] {prompt.split(chr(10))[1]}")
        print(f"  Generated: {script[:100]}")
        if not valid:
            print(f"  Error: {checker.get_errors(script)}")

    rate = checker.syntax_pass_rate(generated_scripts)
    print(f"\nSyntax pass rate: {rate:.1%}")


if __name__ == "__main__":
    main()
