"""Shared helpers for the episode-style pipeline."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Iterable, Iterator


REPO_ROOT = Path(__file__).resolve().parent
GENERATOR_DIR = REPO_ROOT / "example-generator-v1"
FORMATTER_DIR = REPO_ROOT / "raw-to-sft-grpo-dpo-formatter"
GENERATOR_DATA_DIR = GENERATOR_DIR / "data"


def ensure_legacy_paths() -> None:
    """Expose the legacy generator and formatter folders as import roots."""
    for path in (GENERATOR_DIR, FORMATTER_DIR):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def resolve_repo_path(raw_path: str) -> Path:
    """Resolve a user-supplied path against common repo locations."""
    candidate = Path(raw_path)
    if candidate.exists():
        return candidate.resolve()

    search_roots = (REPO_ROOT, GENERATOR_DIR, GENERATOR_DATA_DIR, FORMATTER_DIR)
    for root in search_roots:
        probe = root / raw_path
        if probe.exists():
            return probe.resolve()

    raise FileNotFoundError(f"Could not resolve path: {raw_path}")


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def split_legacy_rendered(rendered_text: str) -> tuple[str, str]:
    """Split a legacy rendered block into question and answer text."""
    marker = "\nCevap:"
    if marker not in rendered_text:
        raise ValueError("Legacy rendered text is missing the expected '\\nCevap:' marker")
    question, answer = rendered_text.split(marker, 1)
    return question.strip(), answer.strip()


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                yield json.loads(stripped)
