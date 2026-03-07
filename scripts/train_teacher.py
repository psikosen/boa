#!/usr/bin/env python3
"""Stage 0: Train the dense teacher baseline.

Usage:
    python scripts/train_teacher.py [--config configs/base.yaml] [--output-dir outputs/stage0]
"""

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from bash_mantis.utils.config import load_config
from bash_mantis.utils.helpers import set_seed, count_parameters, count_parameters_by_group
from bash_mantis.tokenization.byte_tokenizer import ByteTokenizer
from bash_mantis.models.bash_mantis_model import BashMantisModel
from bash_mantis.data.dataset_builders import BashDataset, build_dataset
from bash_mantis.data.synthetic_tasks import SyntheticTaskGenerator
from bash_mantis.training.trainer import BashMantisTrainer


def main():
    parser = argparse.ArgumentParser(description="Train Bash-MANTIS dense teacher")
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--output-dir", type=str, default="outputs/stage0")
    parser.add_argument("--generate-synthetic", action="store_true",
                        help="Generate synthetic training data first")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.training.seed)

    # Generate synthetic data if requested
    if args.generate_synthetic:
        print("Generating synthetic training data...")
        gen = SyntheticTaskGenerator(seed=config.training.seed)
        tasks = gen.generate_all(n_write=500, n_repair=200)
        gen.save_jsonl(tasks, Path(config.data.data_dir) / "nl2bash.jsonl")
        # Also save some as raw bash
        raw_tasks = [{"text": t["bash"]} for t in tasks if "bash" in t]
        gen.save_jsonl(raw_tasks, Path(config.data.data_dir) / "raw_bash.jsonl")
        print(f"  Generated {len(tasks)} tasks")

    # Tokenizer
    tokenizer = ByteTokenizer(config.model.vocab_size)

    # Model
    model = BashMantisModel.from_config(config)
    print(f"Model parameters: {count_parameters(model):,}")
    print("Parameter breakdown:")
    for group, count in count_parameters_by_group(model).items():
        print(f"  {group}: {count:,}")

    # Dataset
    data_dir = Path(config.data.data_dir)
    if data_dir.exists() and any(data_dir.iterdir()):
        dataset = build_dataset(
            data_dir=data_dir,
            tokenizer=tokenizer,
            max_len=config.data.max_seq_len,
        )
        train_loader = DataLoader(dataset, batch_size=config.training.batch_size)
    else:
        print("No training data found. Generating minimal synthetic dataset...")
        gen = SyntheticTaskGenerator(seed=config.training.seed)
        tasks = gen.generate_all(n_write=200, n_repair=50)
        gen.save_jsonl(tasks, data_dir / "nl2bash.jsonl")
        dataset = build_dataset(data_dir=data_dir, tokenizer=tokenizer,
                                max_len=config.data.max_seq_len)
        train_loader = DataLoader(dataset, batch_size=config.training.batch_size)

    # Trainer
    trainer = BashMantisTrainer(
        model=model,
        config=config,
        train_loader=train_loader,
        tokenizer=tokenizer,
        output_dir=args.output_dir,
    )

    # Train
    trainer.train()


if __name__ == "__main__":
    main()
