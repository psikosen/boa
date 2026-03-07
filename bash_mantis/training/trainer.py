"""Trainer for Bash-MANTIS.

Handles the full training loop with:
- Staged curriculum support
- Mixed-precision training
- Gradient clipping
- Logging
- Checkpointing
- Evaluation
"""

from __future__ import annotations

import os
import time
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from bash_mantis.models.bash_mantis_model import BashMantisModel, MantisOutput
from bash_mantis.training.losses import BashMantisLoss, LossOutput
from bash_mantis.training.curriculum import TrainingCurriculum
from bash_mantis.eval.syntax_check import SyntaxChecker
from bash_mantis.eval.metrics import BashMetrics
from bash_mantis.tokenization.byte_tokenizer import ByteTokenizer
from bash_mantis.utils.config import MantisConfig


class BashMantisTrainer:
    """Training loop for Bash-MANTIS."""

    def __init__(
        self,
        model: BashMantisModel,
        config: MantisConfig,
        train_loader: DataLoader,
        eval_loader: DataLoader | None = None,
        teacher_model: BashMantisModel | None = None,
        tokenizer: ByteTokenizer | None = None,
        output_dir: str = "outputs",
    ):
        self.model = model
        self.config = config
        self.tc = config.training
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.teacher_model = teacher_model
        self.tokenizer = tokenizer or ByteTokenizer(config.model.vocab_size)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Device
        self.device = torch.device(
            self.tc.device if torch.cuda.is_available() else "cpu"
        )
        self.model = self.model.to(self.device)
        if self.teacher_model:
            self.teacher_model = self.teacher_model.to(self.device)
            self.teacher_model.eval()

        # Loss
        self.loss_fn = BashMantisLoss.from_config(config)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.tc.learning_rate,
            weight_decay=self.tc.weight_decay,
        )

        # LR scheduler (cosine with warmup)
        self.scheduler = self._build_scheduler()

        # Evaluation
        self.syntax_checker = SyntaxChecker()
        self.metrics = BashMetrics(syntax_checker=self.syntax_checker)

        # Curriculum
        self.curriculum = TrainingCurriculum()

        # State
        self.global_step = 0
        self.best_eval_loss = float("inf")
        self.log_history: list[dict] = []

    def _build_scheduler(self):
        """Build cosine LR scheduler with linear warmup."""

        def lr_lambda(step):
            if step < self.tc.warmup_steps:
                return step / max(1, self.tc.warmup_steps)
            progress = (step - self.tc.warmup_steps) / max(
                1, self.tc.max_steps - self.tc.warmup_steps
            )
            return max(0.1, 0.5 * (1 + __import__("math").cos(3.14159 * progress)))

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def train_step(self, batch: dict[str, torch.Tensor]) -> LossOutput:
        """Execute a single training step."""
        self.model.train()

        input_ids = batch["input_ids"].to(self.device)
        labels = batch["labels"].to(self.device)
        doc_ids = batch.get("doc_ids")
        if doc_ids is not None:
            doc_ids = doc_ids.to(self.device)

        # Forward pass
        output: MantisOutput = self.model(input_ids, doc_ids=doc_ids)

        # Get teacher features if available
        teacher_features = None
        if self.teacher_model is not None and self.tc.lambda_off > 0:
            with torch.no_grad():
                teacher_out = self.teacher_model(input_ids)
                teacher_features = teacher_out.hidden

        # Compute manifold inputs for attractor/contraction
        h_t = output.hidden[:, -1, :]  # last hidden state
        p_bar = self.model.workspace.pool(output.slots)

        # Compute loss
        loss_out = self.loss_fn(
            logits=output.logits,
            targets=labels,
            teacher_features=teacher_features,
            student_features=output.hidden if teacher_features is not None else None,
            kappa=output.kappa,
            manifold=self.model.manifold,
            h_t=h_t,
            p_bar=p_bar,
        )

        # Backward
        self.optimizer.zero_grad()
        loss_out.total.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.tc.grad_clip)
        self.optimizer.step()
        self.scheduler.step()

        return loss_out

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        """Run evaluation on the eval set."""
        if self.eval_loader is None:
            return {}

        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        for batch in self.eval_loader:
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)

            output = self.model(input_ids)
            loss_out = self.loss_fn(logits=output.logits, targets=labels)
            total_loss += loss_out.lm.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        return {"eval_loss": avg_loss}

    def save_checkpoint(self, path: str | Path | None = None):
        """Save model checkpoint."""
        if path is None:
            path = self.output_dir / f"checkpoint_step{self.global_step}.pt"
        path = Path(path)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "global_step": self.global_step,
                "config": {
                    "model": self.config.model.__dict__,
                    "training": self.config.training.__dict__,
                },
            },
            path,
        )

    def load_checkpoint(self, path: str | Path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.global_step = checkpoint["global_step"]

    def train(self):
        """Run the full training loop."""
        print(f"Starting training stage {self.tc.stage}")
        print(f"  Device: {self.device}")
        print(f"  Max steps: {self.tc.max_steps}")
        print(f"  Batch size: {self.tc.batch_size}")
        print(f"  Learning rate: {self.tc.learning_rate}")
        param_count = sum(p.numel() for p in self.model.parameters())
        print(f"  Total parameters: {param_count:,}")

        # Apply ternary if needed
        if self.config.model.ternary_enabled:
            self.model.enable_ternary()
            print("  Ternary quantization: ENABLED")

        data_iter = iter(self.train_loader)
        pbar = tqdm(range(self.global_step, self.tc.max_steps), desc="Training")

        for step in pbar:
            # Get batch
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_loader)
                batch = next(data_iter)

            # Train step
            loss_out = self.train_step(batch)
            self.global_step = step + 1

            # Logging
            if self.global_step % self.tc.log_every == 0:
                lr = self.scheduler.get_last_lr()[0]
                log_entry = {
                    "step": self.global_step,
                    "loss": loss_out.total.item(),
                    "lm_loss": loss_out.lm.item(),
                    "lr": lr,
                }
                self.log_history.append(log_entry)
                pbar.set_postfix(
                    loss=f"{loss_out.total.item():.4f}",
                    lm=f"{loss_out.lm.item():.4f}",
                    lr=f"{lr:.2e}",
                )

            # Evaluation
            if self.global_step % self.tc.eval_every == 0:
                eval_metrics = self.evaluate()
                if eval_metrics:
                    print(f"\n  Step {self.global_step} eval: {eval_metrics}")
                    eval_loss = eval_metrics.get("eval_loss", float("inf"))
                    if eval_loss < self.best_eval_loss:
                        self.best_eval_loss = eval_loss
                        self.save_checkpoint(self.output_dir / "best_model.pt")

            # Checkpointing
            if self.global_step % self.tc.save_every == 0:
                self.save_checkpoint()

        # Final save
        self.save_checkpoint(self.output_dir / "final_model.pt")

        # Save training log
        with open(self.output_dir / "training_log.json", "w") as f:
            json.dump(self.log_history, f, indent=2)

        print(f"\nTraining complete. {self.global_step} steps.")
