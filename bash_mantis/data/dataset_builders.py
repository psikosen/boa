"""Dataset builders for Bash-MANTIS training.

Supports four data sources:
  1. Raw Bash scripts
  2. NL-to-Bash pairs
  3. Bash repair pairs
  4. Bash explanation pairs
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterator

import torch
from torch.utils.data import Dataset, IterableDataset

from bash_mantis.tokenization.byte_tokenizer import ByteTokenizer
from bash_mantis.data.formatters import TaskFormatter


class BashDataset(Dataset):
    """Static dataset for Bash-MANTIS training."""

    def __init__(
        self,
        data_path: str | Path,
        tokenizer: ByteTokenizer,
        max_len: int = 512,
    ):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.samples: list[str] = []

        data_path = Path(data_path)
        if data_path.exists():
            if data_path.suffix == ".jsonl":
                with open(data_path) as f:
                    for line in f:
                        rec = json.loads(line.strip())
                        self.samples.append(rec.get("text", ""))
            elif data_path.suffix == ".json":
                with open(data_path) as f:
                    data = json.load(f)
                    for rec in data:
                        self.samples.append(rec.get("text", ""))
            elif data_path.is_dir():
                for fp in sorted(data_path.glob("*.txt")):
                    self.samples.append(fp.read_text())
                for fp in sorted(data_path.glob("*.sh")):
                    self.samples.append(fp.read_text())

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        text = self.samples[idx]
        token_ids = self.tokenizer.encode(text, add_bos=True, add_eos=True)
        token_ids = self.tokenizer.pad(token_ids, self.max_len)
        input_ids = torch.tensor(token_ids, dtype=torch.long)
        return {"input_ids": input_ids, "labels": input_ids.clone()}


class MixedBashDataset(IterableDataset):
    """Mixed dataset that samples from multiple sources according to ratios.

    Sources:
      - raw_bash: raw shell scripts (40%)
      - nl2bash: NL-to-Bash pairs (30%)
      - repair: Bash repair pairs (15%)
      - explanation: Bash explanation pairs (15%)
    """

    def __init__(
        self,
        data_dir: str | Path,
        tokenizer: ByteTokenizer,
        formatter: TaskFormatter,
        max_len: int = 512,
        raw_bash_ratio: float = 0.40,
        nl2bash_ratio: float = 0.30,
        repair_ratio: float = 0.15,
        explanation_ratio: float = 0.15,
        seed: int = 42,
    ):
        self.tokenizer = tokenizer
        self.formatter = formatter
        self.max_len = max_len
        self.seed = seed

        data_dir = Path(data_dir)

        self.sources: dict[str, list[dict]] = {
            "raw_bash": [],
            "nl2bash": [],
            "repair": [],
            "explanation": [],
        }

        self.ratios = {
            "raw_bash": raw_bash_ratio,
            "nl2bash": nl2bash_ratio,
            "repair": repair_ratio,
            "explanation": explanation_ratio,
        }

        # Load sources if they exist
        for source_name in self.sources:
            path = data_dir / f"{source_name}.jsonl"
            if path.exists():
                with open(path) as f:
                    for line in f:
                        self.sources[source_name].append(json.loads(line.strip()))

    def _format_sample(self, source: str, record: dict) -> str:
        """Format a record from a given source into training text."""
        if source == "raw_bash":
            return record.get("text", record.get("code", ""))
        elif source == "nl2bash":
            intent = record.get("intent", record.get("nl", ""))
            cmd = record.get("bash", record.get("cmd", ""))
            constraints = record.get("constraints", "")
            return self.formatter.format_write(intent, constraints, cmd)
        elif source == "repair":
            intent = record.get("intent", "repair the script")
            broken = record.get("broken", record.get("input", ""))
            fixed = record.get("fixed", record.get("output", ""))
            constraints = record.get("constraints", "")
            return self.formatter.format_fix(intent, broken, constraints, fixed)
        elif source == "explanation":
            script = record.get("script", record.get("code", ""))
            explanation = record.get("explanation", record.get("text", ""))
            return self.formatter.format_explain(script, explanation)
        return ""

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        rng = random.Random(self.seed)
        # Build cumulative distribution
        sources = list(self.ratios.keys())
        weights = [self.ratios[s] for s in sources]
        # Filter to non-empty sources
        available = [(s, w) for s, w in zip(sources, weights) if self.sources[s]]
        if not available:
            return

        names, ws = zip(*available)
        total = sum(ws)
        ws = [w / total for w in ws]

        while True:
            source = rng.choices(names, weights=ws, k=1)[0]
            record = rng.choice(self.sources[source])
            text = self._format_sample(source, record)
            if not text:
                continue
            token_ids = self.tokenizer.encode(text, add_bos=True, add_eos=True)
            token_ids = self.tokenizer.pad(token_ids, self.max_len)
            input_ids = torch.tensor(token_ids, dtype=torch.long)
            yield {"input_ids": input_ids, "labels": input_ids.clone()}


def build_dataset(
    data_dir: str | Path,
    tokenizer: ByteTokenizer,
    max_len: int = 512,
    ratios: dict[str, float] | None = None,
) -> MixedBashDataset:
    """Build a mixed training dataset."""
    formatter = TaskFormatter()
    r = ratios or {}
    return MixedBashDataset(
        data_dir=data_dir,
        tokenizer=tokenizer,
        formatter=formatter,
        max_len=max_len,
        raw_bash_ratio=r.get("raw_bash", 0.40),
        nl2bash_ratio=r.get("nl2bash", 0.30),
        repair_ratio=r.get("repair", 0.15),
        explanation_ratio=r.get("explanation", 0.15),
    )
