# Bash-MANTIS-0.6

**Tiny Ternary Instruction-Following Shell Model**

A ~587K parameter specialized model that maps short English instructions into correct Bash commands through a shared intent-to-execution state space.

## Architecture

Five integrated subsystems:

1. **Ternary Backbone** — 4-layer transformer with DLT-style `{-1, 0, +1}` quantized weights, full-precision embeddings/norms/output head, and tiny precision islands per block
2. **Manifold State** — 12-dim recurrent latent tracking task mode, pipeline state, quoting stability, control flow, repair confidence
3. **Structured Workspace** — 6 persistent slots (command intent, targets, filters, pipes/redirection, variables/control flow, function summary)
4. **Doc-to-LoRA Memory** — Hypernetwork converting short Bash rule cards into rank-2 LoRA adapters for one-pass knowledge internalization
5. **Ternary Preference Head** — Win/tie/lose ranking for candidate Bash outputs, recognizing semantic equivalence

## Target Behavior

- Translate short English intents into Bash
- Complete partial commands
- Write short Bash functions
- Repair broken Bash
- Rank multiple Bash candidates (including recognizing ties)

## Model Config

| Component | Value |
|---|---|
| Tokenizer | Byte-level, vocab 320 |
| Context length | 256 |
| Layers | 4 |
| Model dim | 108 |
| Heads | 4 |
| FFN dim | 256 |
| Manifold dim | 12 |
| Workspace | 6 slots x 24 dims |
| Doc-to-LoRA | Rank-2 adapter |
| Total params | ~587K |

## Prompt Format

```
[TASK] write_bash
[INTENT] find all .csv files larger than 10MB and print their names
[CONSTRAINTS] recursive; safe filenames; no sudo
[OUTPUT]
```

Also supports `fix_bash`, `rank_bash`, `complete_bash`, and `explain_bash` task types.

## Training Curriculum

| Stage | Name | Key Objectives |
|---|---|---|
| 0 | Dense Teacher | LM loss only |
| 1 | Ternary Student | LM + OFF feature distillation |
| 2 | Manifold Warmup | + attractor + contraction loss |
| 3 | Workspace Integration | + workspace slot updates |
| 4 | Instruction Alignment | + text alignment + intent bridge |
| 5 | Document Memory | + Doc-to-LoRA rule cards |
| 6 | Preference Alignment | + TODO win/tie/lose + equivalence clustering |

## Loss Components

1. **LM** — Next-token prediction
2. **OFF** — Teacher-student cosine feature distillation
3. **Text Align** — Paraphrase clustering in latent space
4. **Intent Bridge** — Text instruction to Bash solution latent alignment
5. **Syntax** — `bash -n` validity penalty
6. **Execution** — Sandbox behavioral match (exit code, stdout, stderr, FS delta)
7. **Attractor** — Fixed-point stability for correct terminal states
8. **Contraction** — Jacobian spectral norm regularization
9. **Equivalence** — Behavioral equivalence clustering in latent space
10. **TODO** — Ternary preference (win/tie/lose) alignment

## Quick Start

```bash
# Generate synthetic data and train dense teacher
python scripts/train_teacher.py --config configs/base.yaml --generate-synthetic

# Train ternary student
python scripts/train_student.py --teacher-checkpoint outputs/stage0/final_model.pt

# Full curriculum training
python scripts/train_full.py --stages 0,1,2,3,4,5,6 --generate-synthetic

# Evaluate
python scripts/eval_syntax.py --checkpoint outputs/stage0/final_model.pt
python scripts/eval_exec.py --checkpoint outputs/stage0/final_model.pt
python scripts/eval_latent.py --checkpoint outputs/stage6/final_model.pt
```

## Project Structure

```
bash_mantis/
  tokenization/       # Byte-level tokenizer
  models/
    transformer.py     # RoPE transformer with RMSNorm, SwiGLU
    ternary_linear.py  # DLT quantization + precision islands
    manifold.py        # Recurrent latent state with gated update
    workspace.py       # Persistent slot-based workspace
    lora_memory.py     # Doc encoder + LoRA hypernetwork
    preference_head.py # Win/tie/lose classifier
    bash_mantis_model.py  # Full integrated model
  training/
    losses.py          # All 10 loss components
    curriculum.py      # 7-stage training curriculum
    trainer.py         # Training loop with checkpointing
  eval/
    syntax_check.py    # bash -n validation
    sandbox_exec.py    # Isolated execution with observables
    metrics.py         # Syntax, execution, match metrics
    latent_analysis.py # Manifold geometry analysis
  data/
    formatters.py      # Structured prompt formatting
    dataset_builders.py # Mixed dataset with ratio sampling
    synthetic_tasks.py  # Template-based synthetic generation
configs/               # YAML configs per stage
data/rule_cards/       # Bash reference rule cards for Doc-to-LoRA
scripts/               # Training and evaluation entry points
```

## Requirements

- Python >= 3.10
- PyTorch >= 2.1
- PyYAML, NumPy, tqdm
