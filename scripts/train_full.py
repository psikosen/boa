#!/usr/bin/env python3
"""Train Bash-MANTIS through multiple curriculum stages.

Usage:
    python scripts/train_full.py \
        --base-config configs/base.yaml \
        --stages 0,1,2,3,4,5,6 \
        --output-dir outputs
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
from bash_mantis.data.synthetic_tasks import SyntheticTaskGenerator
from bash_mantis.training.trainer import BashMantisTrainer
from bash_mantis.training.curriculum import TrainingCurriculum


def main():
    parser = argparse.ArgumentParser(description="Full Bash-MANTIS training pipeline")
    parser.add_argument("--base-config", type=str, default="configs/base.yaml")
    parser.add_argument("--stages", type=str, default="0",
                        help="Comma-separated stage numbers to run")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--resume-from", type=str, default=None,
                        help="Checkpoint to resume from")
    parser.add_argument("--generate-synthetic", action="store_true")
    args = parser.parse_args()

    stages = [int(s) for s in args.stages.split(",")]
    config = load_config(args.base_config)
    set_seed(config.training.seed)
    curriculum = TrainingCurriculum()
    tokenizer = ByteTokenizer(config.model.vocab_size)

    # Generate synthetic data if needed
    data_dir = Path(config.data.data_dir)
    if args.generate_synthetic or not data_dir.exists():
        print("Generating synthetic training data...")
        data_dir.mkdir(parents=True, exist_ok=True)
        gen = SyntheticTaskGenerator(seed=config.training.seed)
        tasks = gen.generate_all(n_write=500, n_repair=200)
        gen.save_jsonl(tasks, data_dir / "nl2bash.jsonl")
        raw_tasks = [{"text": t["bash"]} for t in tasks if "bash" in t]
        gen.save_jsonl(raw_tasks, data_dir / "raw_bash.jsonl")

    # Create model
    model = BashMantisModel.from_config(config)
    teacher = None

    # Resume if specified
    if args.resume_from:
        checkpoint = torch.load(args.resume_from, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Resumed from {args.resume_from}")

    for stage_num in stages:
        print(f"\n{'='*60}")
        print(f"STAGE {stage_num}: {curriculum.get_stage(stage_num).name}")
        print(f"{'='*60}")

        # Apply curriculum stage
        config = curriculum.apply_stage(config, stage_num)

        # For stage 1+, use previous stage's model as teacher for OFF
        if stage_num == 1:
            teacher = BashMantisModel.from_config(config)
            prev_path = Path(args.output_dir) / f"stage{stage_num-1}" / "final_model.pt"
            if prev_path.exists():
                ckpt = torch.load(prev_path, map_location="cpu", weights_only=False)
                teacher.load_state_dict(ckpt["model_state_dict"])
                model.load_state_dict(ckpt["model_state_dict"], strict=False)
                teacher.eval()
                for p in teacher.parameters():
                    p.requires_grad_(False)

        # Enable ternary if specified
        if config.model.ternary_enabled:
            model.enable_ternary()

        # Build dataset
        dataset = build_dataset(
            data_dir=config.data.data_dir,
            tokenizer=tokenizer,
            max_len=config.data.max_seq_len,
        )
        train_loader = DataLoader(dataset, batch_size=config.training.batch_size)

        # Train
        stage_dir = Path(args.output_dir) / f"stage{stage_num}"
        trainer = BashMantisTrainer(
            model=model,
            config=config,
            train_loader=train_loader,
            teacher_model=teacher,
            tokenizer=tokenizer,
            output_dir=str(stage_dir),
        )
        trainer.train()

        print(f"Stage {stage_num} complete. Model saved to {stage_dir}")

    print(f"\nAll stages complete. Parameters: {count_parameters(model):,}")


if __name__ == "__main__":
    main()
