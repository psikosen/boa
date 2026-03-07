"""Task formatters for Bash-MANTIS structured prompts.

Formats tasks into the canonical prompt format:
  [TASK] write_bash / fix_bash / rank_bash
  [INTENT] ...
  [CONSTRAINTS] ...
  [OUTPUT] ...
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BashTask:
    """A structured Bash task."""
    task_type: str  # write_bash, fix_bash, rank_bash, complete_bash, explain_bash
    intent: str
    constraints: str = ""
    input_script: str = ""
    candidates: list[str] | None = None
    output: str = ""


class TaskFormatter:
    """Formats tasks into structured prompt strings."""

    def format_write(self, intent: str, constraints: str = "", output: str = "") -> str:
        """Format a write_bash task."""
        parts = [f"[TASK] write_bash", f"[INTENT] {intent}"]
        if constraints:
            parts.append(f"[CONSTRAINTS] {constraints}")
        parts.append("[OUTPUT]")
        if output:
            parts.append(output)
        return "\n".join(parts)

    def format_fix(
        self, intent: str, input_script: str, constraints: str = "", output: str = ""
    ) -> str:
        """Format a fix_bash task."""
        parts = [
            f"[TASK] fix_bash",
            f"[INTENT] {intent}",
            f"[INPUT_SCRIPT]\n{input_script}",
        ]
        if constraints:
            parts.append(f"[CONSTRAINTS] {constraints}")
        parts.append("[OUTPUT]")
        if output:
            parts.append(output)
        return "\n".join(parts)

    def format_rank(
        self, intent: str, candidates: list[str], output: str = ""
    ) -> str:
        """Format a rank_bash task."""
        parts = [f"[TASK] rank_bash", f"[INTENT] {intent}"]
        for i, c in enumerate(candidates, 1):
            parts.append(f"[CANDIDATE_{i}] {c}")
        parts.append("[OUTPUT]")
        if output:
            parts.append(output)
        return "\n".join(parts)

    def format_complete(self, partial: str, constraints: str = "", output: str = "") -> str:
        """Format a complete_bash task."""
        parts = [
            f"[TASK] complete_bash",
            f"[INTENT] complete the command",
            f"[INPUT_SCRIPT]\n{partial}",
        ]
        if constraints:
            parts.append(f"[CONSTRAINTS] {constraints}")
        parts.append("[OUTPUT]")
        if output:
            parts.append(output)
        return "\n".join(parts)

    def format_explain(self, script: str, output: str = "") -> str:
        """Format an explain_bash task."""
        parts = [
            f"[TASK] explain_bash",
            f"[INTENT] explain the script",
            f"[INPUT_SCRIPT]\n{script}",
            "[OUTPUT]",
        ]
        if output:
            parts.append(output)
        return "\n".join(parts)

    def format_task(self, task: BashTask) -> str:
        """Format any BashTask into a prompt string."""
        if task.task_type == "write_bash":
            return self.format_write(task.intent, task.constraints, task.output)
        elif task.task_type == "fix_bash":
            return self.format_fix(task.intent, task.input_script, task.constraints, task.output)
        elif task.task_type == "rank_bash":
            return self.format_rank(task.intent, task.candidates or [], task.output)
        elif task.task_type == "complete_bash":
            return self.format_complete(task.input_script, task.constraints, task.output)
        elif task.task_type == "explain_bash":
            return self.format_explain(task.input_script, task.output)
        else:
            raise ValueError(f"Unknown task type: {task.task_type}")

    def parse_output(self, text: str) -> str:
        """Extract the output portion from a formatted prompt."""
        marker = "[OUTPUT]"
        idx = text.find(marker)
        if idx == -1:
            return text
        return text[idx + len(marker) :].strip()
