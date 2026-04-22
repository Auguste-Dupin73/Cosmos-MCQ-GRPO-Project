from __future__ import annotations

import json
import os
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Iterable, Mapping, Sequence

import yaml
from transformers import TrainerCallback

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


REPO_ROOT = Path(__file__).resolve().parents[1]


def resolve_input_path(raw_path: str | Path, base_dir: str | Path | None = None) -> Path:
    candidate = Path(raw_path)
    if candidate.is_absolute() and candidate.exists():
        return candidate.resolve()

    search_roots: list[Path] = []
    if base_dir is not None:
        search_roots.append(Path(base_dir))
    search_roots.extend([Path.cwd(), REPO_ROOT])

    for root in search_roots:
        probe = (root / candidate).resolve()
        if probe.exists():
            return probe

    raise FileNotFoundError(f"Could not resolve path: {raw_path}")


def resolve_output_path(raw_path: str | Path) -> Path:
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate
    return (REPO_ROOT / candidate).resolve()


def coerce_path_list(value: str | Path | Sequence[str | Path] | None) -> list[Path]:
    if value is None:
        return []
    if isinstance(value, (str, Path)):
        return [resolve_input_path(value)]
    return [resolve_input_path(item) for item in value]


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    config_path = resolve_input_path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file must decode to a mapping: {config_path}")
    return data


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def set_seed(seed: int) -> None:
    random.seed(seed)
    if np is not None:
        np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def extract_tag_value(tags: Sequence[str] | None, prefix: str, default: str | None = None) -> str | None:
    if not tags:
        return default
    marker = f"{prefix}:"
    for tag in tags:
        if isinstance(tag, str) and tag.startswith(marker):
            return tag[len(marker) :]
    return default


def build_common_metadata(row: Mapping[str, Any], source_path: str | Path) -> dict[str, Any]:
    tags = list(row.get("tags", []) or [])
    return {
        "id": row.get("episode_id") or row.get("id"),
        "tags": tags,
        "skill_id": row.get("skill_id") or extract_tag_value(tags, "skill"),
        "template_id": row.get("template_id") or extract_tag_value(tags, "template"),
        "tier": row.get("tier") or extract_tag_value(tags, "tier"),
        "difficulty": row.get("difficulty") or extract_tag_value(tags, "difficulty"),
        "language": row.get("language") or extract_tag_value(tags, "lang"),
        "source_path": str(Path(source_path)),
    }


def take_limit(rows: list[dict[str, Any]], max_samples: int | None) -> list[dict[str, Any]]:
    if max_samples is None or max_samples < 0:
        return rows
    return rows[:max_samples]


def split_records(
    rows: Sequence[dict[str, Any]],
    eval_size: float | int | None,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = list(rows)
    if not rows or eval_size in (None, 0, 0.0):
        return rows, []

    if isinstance(eval_size, float):
        if not 0.0 < eval_size < 1.0:
            raise ValueError("Float eval_size must be between 0 and 1")
        eval_count = max(1, int(round(len(rows) * eval_size)))
    else:
        eval_count = int(eval_size)
    eval_count = min(max(eval_count, 1), len(rows) - 1)

    rng = random.Random(seed)
    shuffled = list(rows)
    rng.shuffle(shuffled)
    eval_rows = shuffled[:eval_count]
    train_rows = shuffled[eval_count:]
    return train_rows, eval_rows


def latest_checkpoint(output_dir: str | Path) -> Path | None:
    root = Path(output_dir)
    if not root.exists():
        return None
    checkpoints = sorted(
        (path for path in root.glob("checkpoint-*") if path.is_dir()),
        key=lambda path: int(path.name.split("-")[-1]),
    )
    return checkpoints[-1] if checkpoints else None


def coerce_torch_dtype(value: Any) -> Any:
    if value is None or torch is None:
        return value
    if not isinstance(value, str):
        return value
    lowered = value.lower().strip()
    if lowered == "auto":
        return "auto"
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    return mapping.get(lowered, value)


def _safe_mean(values: Sequence[float]) -> float:
    return float(mean(values)) if values else 0.0


def _safe_std(values: Sequence[float]) -> float:
    return float(pstdev(values)) if len(values) > 1 else 0.0


DEFAULT_METRIC_FIELDS = (
    "reward",
    "main_accuracy",
    "option_accuracy",
    "probe_accuracy",
    "joint_success",
    "correct_option_wrong_reasoning",
    "reasoning_consistent",
)


def summarize_metric_rows(
    rows: Sequence[Mapping[str, Any]],
    metric_fields: Sequence[str] = DEFAULT_METRIC_FIELDS,
) -> dict[str, Any]:
    summary: dict[str, Any] = {"count": len(rows)}
    for field in metric_fields:
        values = [float(row[field]) for row in rows if row.get(field) is not None]
        if not values:
            continue
        if field == "reward":
            summary["reward_mean"] = _safe_mean(values)
            summary["reward_std"] = _safe_std(values)
        else:
            summary[field] = _safe_mean(values)
    return summary


def summarize_metric_groups(
    rows: Sequence[Mapping[str, Any]],
    group_field: str,
    metric_fields: Sequence[str] = DEFAULT_METRIC_FIELDS,
) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        key = str(row.get(group_field) or "unknown")
        grouped[key].append(row)
    return {
        key: summarize_metric_rows(group_rows, metric_fields=metric_fields)
        for key, group_rows in sorted(grouped.items())
    }


class ProgressMetricsCallback(TrainerCallback):
    """Add a simple samples_seen counter to Trainer logs."""

    def __init__(self, num_generations: int = 1) -> None:
        self.num_generations = max(int(num_generations or 1), 1)

    def on_log(self, args, state, control, logs=None, **kwargs):  # type: ignore[override]
        if logs is None:
            return control
        world_size = int(getattr(args, "world_size", 0) or os.environ.get("WORLD_SIZE", "1"))
        prompts_per_step = int(args.per_device_train_batch_size) * int(args.gradient_accumulation_steps) * max(world_size, 1)
        logs.setdefault("step", state.global_step)
        logs["samples_seen"] = state.global_step * prompts_per_step
        logs["completion_samples_seen"] = logs["samples_seen"] * self.num_generations
        return control
