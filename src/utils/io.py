"""JSONL read/write helpers.

Centralises I/O so every pipeline stage serialises consistently.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    """Write a list of dicts as newline-delimited JSON.

    Creates parent directories if they don't exist.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file and return a list of dicts.

    Raises FileNotFoundError if *path* does not exist.
    """
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
