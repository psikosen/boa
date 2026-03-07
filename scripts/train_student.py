#!/usr/bin/env python3
"""Stage 1: Train the ternary student with OFF distillation.

Usage:
    python scripts/train_student.py \
        --config configs/ternary.yaml \
        --teacher-checkpoint outputs/stage0/final_model.pt \
        --output-dir outputs/stage1
"""

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from bash_mantis.utils.config import load_config
from bash_mantis.utils.helpers import set_seed, count_parameters
from bash_mantis.tokenization.byte_tokenizer import ByteTokenizer
from bash_mantis.models.bash_mantis_model import BashMantisModel
from bash_mantis.data.dataset_builders import build_dataset
from bash_mantis.training.trainer import BashMantisTrainer


def main():
    parser = argparse.ArgumentParser(description="Train Bash-MANTIS ternary student")
    parser.add_argument("--config", type=str, default="configs/ternary.yaml")
    parser.add_argument("--teacher-checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="outputs/stage1")
    args = parser.parse_args()

    # Load base config then override with ternary config
    base_config = load_config("configs/base.yaml")
    ternary_config = load_config(args.config)

    # Merge: ternary overrides base
    config = base_config
    for field in ["ternary_enabled", "precision_island_size"]:
        if hasattr(ternary_config.model, field):
            setattr(config.model, field, getattr(ternary_config.model, field))
    for field in vars(ternary_config.training):
        val = getattr(ternary_config.training, field)
        setattr(config.training, field, val)

    set_seed(config.training.seed)

    tokenizer = ByteTokenizer(config.model.vocab_size)

    # Student model (ternary)
    student = BashMantisModel.from_config(config)
    print(f"Student parameters: {count_parameters(student):,}")

    # Teacher model (dense, frozen)
    teacher_config = load_config("configs/base.yaml")
    teacher = BashMantisModel.from_config(teacher_config)
    checkpoint = torch.load(args.teacher_checkpoint, map_location="cpu", weights_only=False)
    teacher.load_state_dict(checkpoint["model_state_dict"])
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    print(f"Teacher loaded from {args.teacher_checkpoint}")

    # Initialize student from teacher weights
    student.load_state_dict(checkpoint["model_state_dict"], strict=False)
    print("Student initialized from teacher weights")

    # Dataset
    dataset = build_dataset(
        data_dir=config.data.data_dir,
        tokenizer=tokenizer,
        max_len=config.data.max_seq_len,
    )
    train_loader = DataLoader(dataset, batch_size=config.training.batch_size)

    # Train
    trainer = BashMantisTrainer(
        model=student,
        config=config,
        train_loader=train_loader,
        teacher_model=teacher,
        tokenizer=tokenizer,
        output_dir=args.output_dir,
    )
    trainer.train()


if __name__ == "__main__":
    main()
