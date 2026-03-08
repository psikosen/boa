"""Synthetic task generator for Bash-MANTIS training data.

Generates synthetic NL-to-Bash, repair, and explanation training pairs
from templates and composition rules.
"""

from __future__ import annotations

import random
import json
from dataclasses import dataclass
from pathlib import Path

from bash_mantis.data.formatters import TaskFormatter, BashTask


# Template families for synthetic generation
FILE_DISCOVERY_TEMPLATES = [
    {
        "intent": "find all {ext} files in the current directory",
        "bash": 'find . -type f -name "*.{ext}"',
        "ext_options": ["log", "txt", "csv", "json", "xml", "yaml", "py", "sh", "conf"],
    },
    {
        "intent": "find all {ext} files larger than {size}",
        "bash": 'find . -type f -name "*.{ext}" -size +{size}',
        "ext_options": ["log", "txt", "csv", "json"],
        "size_options": ["1M", "10M", "50M", "100M", "1G"],
    },
    {
        "intent": "find files modified in the last {days} days",
        "bash": "find . -type f -mtime -{days}",
        "days_options": ["1", "3", "7", "14", "30"],
    },
]

PIPELINE_TEMPLATES = [
    {
        "intent": "count the number of lines in all {ext} files",
        "bash": 'find . -name "*.{ext}" -exec wc -l {{}} + | sort -n',
        "ext_options": ["txt", "csv", "log", "py", "sh"],
    },
    {
        "intent": "find unique values in column {col} of {file}",
        "bash": "cut -d',' -f{col} {file} | sort | uniq",
        "col_options": ["1", "2", "3"],
        "file_options": ["data.csv", "input.csv", "records.csv"],
    },
    {
        "intent": "sort {file} by the {col}th field and remove duplicates",
        "bash": "sort -t',' -k{col},{col} {file} | uniq",
        "col_options": ["1", "2", "3"],
        "file_options": ["data.csv", "input.txt"],
    },
]

LOOP_TEMPLATES = [
    {
        "intent": "compress all {ext} files in the current directory",
        "bash": 'for f in *.{ext}; do gzip "$f"; done',
        "ext_options": ["log", "txt", "csv"],
    },
    {
        "intent": "rename all {ext1} files to {ext2}",
        "bash": 'for f in *.{ext1}; do mv "$f" "${{f%.{ext1}}}.{ext2}"; done',
        "ext1_options": ["txt", "log", "dat"],
        "ext2_options": ["bak", "old", "archive"],
    },
]

FUNCTION_TEMPLATES = [
    {
        "intent": "write a function that retries a command up to {n} times",
        "bash": 'retry() {{\n  local n={n}\n  local cmd="$@"\n  for ((i=1; i<=n; i++)); do\n    $cmd && return 0\n    sleep 1\n  done\n  return 1\n}}',
        "n_options": ["3", "5", "10"],
    },
    {
        "intent": "write a function that creates a timestamped backup of a file",
        "bash": 'backup() {{\n  local file="$1"\n  local ts\n  ts=$(date +%Y%m%d_%H%M%S)\n  cp "$file" "${{file}}.${{ts}}.bak"\n}}',
    },
]

SCRIPT_TEMPLATES = [
    {
        "intent": "write a script that deletes {ext} files older than {days} days with confirmation",
        "bash": (
            '#!/bin/bash\nset -euo pipefail\n\n'
            'dir="${{1:-.}}"\n'
            'files=$(find "$dir" -type f -name "*.{ext}" -mtime +{days})\n'
            'count=$(echo "$files" | grep -c . || true)\n\n'
            'if [ "$count" -eq 0 ]; then\n'
            '  echo "No .{ext} files older than {days} days."\n'
            '  exit 0\n'
            'fi\n\n'
            'echo "Found $count files:"\n'
            'echo "$files" | head -10\n\n'
            'read -rp "Delete? [y/N] " ans\n'
            'if [[ "$ans" =~ ^[Yy]$ ]]; then\n'
            '  echo "$files" | xargs -d\'\\n\' rm -v\n'
            'fi'
        ),
        "ext_options": ["log", "tmp", "bak", "old"],
        "days_options": ["7", "14", "30", "90"],
    },
    {
        "intent": "write a script that monitors disk usage and warns if any partition exceeds {pct}%",
        "bash": (
            '#!/bin/bash\nset -euo pipefail\n\n'
            'threshold={pct}\n\n'
            'df -h --output=pcent,target | tail -n +2 | while read -r usage mount; do\n'
            '  pct="${{usage%%%}}"\n'
            '  if [ "$pct" -ge "$threshold" ]; then\n'
            '    echo "WARNING: $mount at ${{usage}}" >&2\n'
            '  fi\n'
            'done'
        ),
        "pct_options": ["80", "85", "90", "95"],
    },
    {
        "intent": "write a function that archives a directory to a tar.gz with error handling",
        "bash": (
            'archive_dir() {{\n'
            '  local src="$1"\n'
            '  local dest="${{2:-$(basename "$src")-$(date +%Y%m%d).tar.gz}}"\n\n'
            '  [ ! -d "$src" ] && {{ echo "Error: not a directory" >&2; return 1; }}\n'
            '  [ -f "$dest" ] && {{ echo "Error: $dest exists" >&2; return 1; }}\n\n'
            '  if tar -czf "$dest" -C "$(dirname "$src")" "$(basename "$src")"; then\n'
            '    echo "Done: $(du -h "$dest" | cut -f1)"\n'
            '  else\n'
            '    rm -f "$dest"\n'
            '    return 1\n'
            '  fi\n'
            '}}'
        ),
    },
    {
        "intent": "write a function that parses -n name, -o outdir, and -v verbose flags",
        "bash": (
            'parse_args() {{\n'
            '  local name="" outdir="." verbose=0\n\n'
            '  while [ $# -gt 0 ]; do\n'
            '    case "$1" in\n'
            '      -n|--name)   name="$2"; shift 2 ;;\n'
            '      -o|--output) outdir="$2"; shift 2 ;;\n'
            '      -v|--verbose) verbose=1; shift ;;\n'
            '      *) echo "Unknown: $1" >&2; return 1 ;;\n'
            '    esac\n'
            '  done\n\n'
            '  [ -z "$name" ] && {{ echo "--name required" >&2; return 1; }}\n'
            '  echo "$name" "$outdir" "$verbose"\n'
            '}}'
        ),
    },
]

REPAIR_TEMPLATES = [
    {
        "intent": "fix the script so spaces in filenames do not break it",
        "broken": 'for f in $(find . -name "*.txt"); do\n  wc -l $f\ndone',
        "fixed": 'find . -name "*.txt" -print0 | while IFS= read -r -d \'\' f; do\n  wc -l "$f"\ndone',
    },
    {
        "intent": "fix the quoting in this command",
        "broken": "echo $HOME/$USER is the home directory",
        "fixed": 'echo "$HOME/$USER is the home directory"',
    },
    {
        "intent": "fix the pipe error handling",
        "broken": "cat file.txt | grep pattern | sort",
        "fixed": 'grep pattern file.txt | sort || echo "Error in pipeline" >&2',
    },
    {
        "intent": "fix the deploy function to handle missing arguments and quote variables",
        "broken": (
            'deploy() {\n'
            '  scp $1 $2:$3\n'
            '  ssh $2 "cd $3 && ./restart.sh"\n'
            '  echo deployed\n'
            '}'
        ),
        "fixed": (
            'deploy() {\n'
            '  local file="${1:?Usage: deploy FILE HOST DIR}"\n'
            '  local host="${2:?}" dir="${3:?}"\n\n'
            '  [ ! -f "$file" ] && { echo "Not found: $file" >&2; return 1; }\n'
            '  scp "$file" "$host":"$dir"/ || return 1\n'
            '  ssh "$host" "cd \'$dir\' && ./restart.sh"\n'
            '}'
        ),
    },
    {
        "intent": "fix the log rotation script that fails on empty directories",
        "broken": (
            '#!/bin/bash\n'
            'cd /var/log/app\n'
            'for f in *.log; do\n'
            '  gzip $f\n'
            '  mv $f.gz archive/\n'
            'done'
        ),
        "fixed": (
            '#!/bin/bash\nset -euo pipefail\n\n'
            'logdir="/var/log/app"\n'
            'archive="$logdir/archive"\n'
            'mkdir -p "$archive"\n\n'
            'shopt -s nullglob\n'
            'for f in "$logdir"/*.log; do\n'
            '  gzip "$f"\n'
            '  mv "$f.gz" "$archive"/\n'
            'done\n'
            'shopt -u nullglob'
        ),
    },
]


class SyntheticTaskGenerator:
    """Generates synthetic Bash training data from templates."""

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.formatter = TaskFormatter()

    def _fill_template(self, template: dict) -> dict:
        """Fill a template with random option values."""
        result = dict(template)
        for key in list(result.keys()):
            if key.endswith("_options"):
                param = key[: -len("_options")]
                chosen = self.rng.choice(result[key])
                # Replace in all string fields
                for field in ["intent", "bash", "broken", "fixed"]:
                    if field in result:
                        result[field] = result[field].replace(f"{{{param}}}", chosen)
                del result[key]
        return result

    def generate_write_tasks(self, n: int = 100) -> list[dict]:
        """Generate NL-to-Bash write tasks."""
        all_templates = (
            FILE_DISCOVERY_TEMPLATES + PIPELINE_TEMPLATES + LOOP_TEMPLATES
            + FUNCTION_TEMPLATES + SCRIPT_TEMPLATES
        )
        tasks = []
        for _ in range(n):
            template = self.rng.choice(all_templates)
            filled = self._fill_template(template)
            text = self.formatter.format_write(
                intent=filled["intent"],
                output=filled["bash"],
            )
            tasks.append({"text": text, "type": "nl2bash",
                          "intent": filled["intent"], "bash": filled["bash"]})
        return tasks

    def generate_repair_tasks(self, n: int = 50) -> list[dict]:
        """Generate Bash repair tasks."""
        tasks = []
        for _ in range(n):
            template = self.rng.choice(REPAIR_TEMPLATES)
            filled = self._fill_template(template)
            text = self.formatter.format_fix(
                intent=filled["intent"],
                input_script=filled["broken"],
                output=filled["fixed"],
            )
            tasks.append({"text": text, "type": "repair",
                          "intent": filled["intent"],
                          "broken": filled["broken"],
                          "fixed": filled["fixed"]})
        return tasks

    def generate_all(self, n_write: int = 200, n_repair: int = 100) -> list[dict]:
        """Generate a mixed set of synthetic tasks."""
        tasks = []
        tasks.extend(self.generate_write_tasks(n_write))
        tasks.extend(self.generate_repair_tasks(n_repair))
        self.rng.shuffle(tasks)
        return tasks

    def save_jsonl(self, tasks: list[dict], output_path: str | Path) -> None:
        """Save tasks as JSONL."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            for task in tasks:
                f.write(json.dumps(task) + "\n")
