"""Configuration loading and management for Bash-MANTIS."""

from __future__ import annotations

import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ModelConfig:
    vocab_size: int = 320
    d_model: int = 108
    n_heads: int = 4
    n_layers: int = 4
    ffn_dim: int = 256
    ctx_len: int = 256
    dropout: float = 0.1
    # Manifold
    manifold_dim: int = 12
    # Workspace
    workspace_slots: int = 6
    workspace_slot_dim: int = 24
    # Doc-to-LoRA
    lora_rank: int = 2
    doc_encoder_dim: int = 64
    doc_max_len: int = 128
    lora_target_block: int = 2  # which transformer block gets the LoRA
    # Preference head
    pref_hidden_dim: int = 48
    # Precision policy
    ternary_enabled: bool = False
    precision_island_size: int = 16  # full-precision units per block


@dataclass
class TrainingConfig:
    # General
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    max_steps: int = 50000
    warmup_steps: int = 1000
    grad_clip: float = 1.0
    seed: int = 42
    eval_every: int = 500
    save_every: int = 2000
    log_every: int = 50
    device: str = "cuda"
    # Loss weights
    lambda_lm: float = 1.0
    lambda_off: float = 0.0
    lambda_text_align: float = 0.0
    lambda_intent_bridge: float = 0.0
    lambda_syntax: float = 0.0
    lambda_exec: float = 0.0
    lambda_attract: float = 0.0
    lambda_contract: float = 0.0
    lambda_equiv: float = 0.0
    lambda_todo: float = 0.0
    # Contraction
    contraction_gamma: float = 0.95
    equiv_margin: float = 1.0
    equiv_beta: float = 0.5
    # Curriculum stage
    stage: int = 0


@dataclass
class DataConfig:
    data_dir: str = "data/processed"
    rule_cards_dir: str = "data/rule_cards"
    max_seq_len: int = 256
    # Data mixture ratios
    raw_bash_ratio: float = 0.40
    nl2bash_ratio: float = 0.30
    repair_ratio: float = 0.15
    explanation_ratio: float = 0.15


@dataclass
class SandboxConfig:
    enabled: bool = False
    temp_dir: str = "/tmp/mantis_sandbox"
    timeout_seconds: int = 5
    max_memory_mb: int = 64
    allowed_binaries: list[str] = field(default_factory=lambda: [
        "bash", "find", "grep", "sed", "awk", "sort", "uniq", "cut", "tr",
        "wc", "tar", "gzip", "xargs", "cat", "printf", "echo", "mv", "cp",
        "rm", "mkdir", "touch", "head", "tail",
    ])


@dataclass
class MantisConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    sandbox: SandboxConfig = field(default_factory=SandboxConfig)


def _apply_dict(dc: Any, d: dict) -> None:
    """Apply a dictionary of overrides to a dataclass instance."""
    for k, v in d.items():
        if hasattr(dc, k):
            setattr(dc, k, v)


def load_config(path: str | Path) -> MantisConfig:
    """Load a YAML config file and return a MantisConfig."""
    path = Path(path)
    cfg = MantisConfig()
    if path.exists():
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        if "model" in raw:
            _apply_dict(cfg.model, raw["model"])
        if "training" in raw:
            _apply_dict(cfg.training, raw["training"])
        if "data" in raw:
            _apply_dict(cfg.data, raw["data"])
        if "sandbox" in raw:
            _apply_dict(cfg.sandbox, raw["sandbox"])
    return cfg
