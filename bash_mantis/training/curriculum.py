"""Training curriculum for Bash-MANTIS staged training.

Stages:
  0: Dense teacher — LM loss only
  1: Ternary student — LM + OFF
  2: Manifold warmup — add attractor + contraction
  3: Workspace integration — add workspace with manifold
  4: Instruction alignment — add text_align + intent_bridge
  5: Document memory — add Doc-to-LoRA
  6: Preference alignment — add TODO + equivalence
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class StageConfig:
    """Configuration for a curriculum stage."""
    stage: int
    name: str
    ternary_enabled: bool
    lambda_lm: float
    lambda_off: float
    lambda_text_align: float
    lambda_intent_bridge: float
    lambda_syntax: float
    lambda_exec: float
    lambda_attract: float
    lambda_contract: float
    lambda_equiv: float
    lambda_todo: float
    learning_rate: float
    max_steps: int


# Default curriculum
STAGES = [
    StageConfig(
        stage=0, name="dense_teacher",
        ternary_enabled=False,
        lambda_lm=1.0, lambda_off=0.0, lambda_text_align=0.0,
        lambda_intent_bridge=0.0, lambda_syntax=0.0, lambda_exec=0.0,
        lambda_attract=0.0, lambda_contract=0.0, lambda_equiv=0.0,
        lambda_todo=0.0, learning_rate=3e-4, max_steps=50000,
    ),
    StageConfig(
        stage=1, name="ternary_student",
        ternary_enabled=True,
        lambda_lm=1.0, lambda_off=0.5, lambda_text_align=0.0,
        lambda_intent_bridge=0.0, lambda_syntax=0.0, lambda_exec=0.0,
        lambda_attract=0.0, lambda_contract=0.0, lambda_equiv=0.0,
        lambda_todo=0.0, learning_rate=2e-4, max_steps=30000,
    ),
    StageConfig(
        stage=2, name="manifold_warmup",
        ternary_enabled=True,
        lambda_lm=1.0, lambda_off=0.3, lambda_text_align=0.0,
        lambda_intent_bridge=0.0, lambda_syntax=0.1, lambda_exec=0.0,
        lambda_attract=0.1, lambda_contract=0.05, lambda_equiv=0.0,
        lambda_todo=0.0, learning_rate=1.5e-4, max_steps=20000,
    ),
    StageConfig(
        stage=3, name="workspace_integration",
        ternary_enabled=True,
        lambda_lm=1.0, lambda_off=0.3, lambda_text_align=0.0,
        lambda_intent_bridge=0.0, lambda_syntax=0.1, lambda_exec=0.0,
        lambda_attract=0.1, lambda_contract=0.05, lambda_equiv=0.0,
        lambda_todo=0.0, learning_rate=1.5e-4, max_steps=20000,
    ),
    StageConfig(
        stage=4, name="instruction_alignment",
        ternary_enabled=True,
        lambda_lm=1.0, lambda_off=0.2, lambda_text_align=0.1,
        lambda_intent_bridge=0.15, lambda_syntax=0.1, lambda_exec=0.0,
        lambda_attract=0.1, lambda_contract=0.05, lambda_equiv=0.0,
        lambda_todo=0.0, learning_rate=1e-4, max_steps=20000,
    ),
    StageConfig(
        stage=5, name="document_memory",
        ternary_enabled=True,
        lambda_lm=1.0, lambda_off=0.2, lambda_text_align=0.1,
        lambda_intent_bridge=0.15, lambda_syntax=0.15, lambda_exec=0.1,
        lambda_attract=0.1, lambda_contract=0.05, lambda_equiv=0.0,
        lambda_todo=0.0, learning_rate=1e-4, max_steps=20000,
    ),
    StageConfig(
        stage=6, name="preference_alignment",
        ternary_enabled=True,
        lambda_lm=1.0, lambda_off=0.2, lambda_text_align=0.1,
        lambda_intent_bridge=0.15, lambda_syntax=0.15, lambda_exec=0.2,
        lambda_attract=0.1, lambda_contract=0.05, lambda_equiv=0.1,
        lambda_todo=0.15, learning_rate=1e-4, max_steps=40000,
    ),
]


class TrainingCurriculum:
    """Manages the staged training curriculum."""

    def __init__(self, stages: list[StageConfig] | None = None):
        self.stages = stages or STAGES

    def get_stage(self, stage_num: int) -> StageConfig:
        """Get config for a specific stage."""
        for s in self.stages:
            if s.stage == stage_num:
                return s
        raise ValueError(f"Stage {stage_num} not found")

    def get_all_stages(self) -> list[StageConfig]:
        """Get all stages in order."""
        return sorted(self.stages, key=lambda s: s.stage)

    def apply_stage(self, config, stage_num: int):
        """Apply a curriculum stage to a MantisConfig."""
        stage = self.get_stage(stage_num)

        config.model.ternary_enabled = stage.ternary_enabled
        config.training.stage = stage.stage
        config.training.learning_rate = stage.learning_rate
        config.training.max_steps = stage.max_steps
        config.training.lambda_lm = stage.lambda_lm
        config.training.lambda_off = stage.lambda_off
        config.training.lambda_text_align = stage.lambda_text_align
        config.training.lambda_intent_bridge = stage.lambda_intent_bridge
        config.training.lambda_syntax = stage.lambda_syntax
        config.training.lambda_exec = stage.lambda_exec
        config.training.lambda_attract = stage.lambda_attract
        config.training.lambda_contract = stage.lambda_contract
        config.training.lambda_equiv = stage.lambda_equiv
        config.training.lambda_todo = stage.lambda_todo

        return config
