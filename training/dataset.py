from __future__ import annotations

import random
from typing import Any, Mapping, Sequence

from datasets import Dataset

from training.formatting import build_episode_prompt, build_task_prompt, derive_episode_targets_from_text, extract_prompt_options
from training.utils import build_common_metadata, read_jsonl, resolve_input_path, take_limit


SOURCE_EPISODE = "source_episode"
SOURCE_EPISODE_SPLIT = "source_episode_split"
EPISODE_GRPO_OFFLINE = "episode_grpo_offline"
EPISODE_GRPO_ONLINE = "episode_grpo_online"


def detect_dataset_format(row: Mapping[str, Any]) -> str:
    if "episode_id" in row and "main" in row and "probe" in row:
        return SOURCE_EPISODE
    if "responses" in row:
        return EPISODE_GRPO_OFFLINE
    if "gold" in row:
        return EPISODE_GRPO_ONLINE
    raise ValueError("Could not detect dataset format from row keys")


def load_episode_records(
    paths: str | Sequence[str],
    *,
    dataset_format: str = "auto",
    include_support_pack: bool = True,
    append_response_format: bool = False,
    split_main_probe: bool = False,
    max_samples: int | None = None,
    shuffle: bool = False,
    seed: int = 42,
) -> list[dict[str, Any]]:
    path_list = [paths] if isinstance(paths, str) else list(paths)
    records: list[dict[str, Any]] = []
    seen_formats: set[str] = set()

    for raw_path in path_list:
        path = resolve_input_path(raw_path)
        rows = read_jsonl(path)
        if not rows:
            continue
        current_format = detect_dataset_format(rows[0]) if dataset_format == "auto" else dataset_format
        seen_formats.add(current_format)
        source_split = split_main_probe or current_format == SOURCE_EPISODE_SPLIT
        row_format = SOURCE_EPISODE if current_format == SOURCE_EPISODE_SPLIT else current_format
        for row in rows:
            records.extend(
                _convert_rows(
                    row,
                    path=str(path),
                    dataset_format=row_format,
                    include_support_pack=include_support_pack,
                    append_response_format=append_response_format,
                    split_main_probe=source_split,
                )
            )

    if len(seen_formats) > 1:
        raise ValueError(f"Mixed dataset formats are not supported in one load: {sorted(seen_formats)}")

    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(records)
    return take_limit(records, max_samples)


def to_hf_dataset(records: Sequence[Mapping[str, Any]]) -> Dataset:
    if not records:
        raise ValueError("No dataset rows were loaded")
    return Dataset.from_list([dict(record) for record in records])


def _convert_rows(
    row: Mapping[str, Any],
    *,
    path: str,
    dataset_format: str,
    include_support_pack: bool,
    append_response_format: bool,
    split_main_probe: bool,
) -> list[dict[str, Any]]:
    if dataset_format == SOURCE_EPISODE:
        if split_main_probe:
            return _convert_source_episode_split_rows(
                row,
                path=path,
                include_support_pack=include_support_pack,
                append_response_format=append_response_format,
            )
        return [
            _convert_source_episode_row(
                row,
                path=path,
                include_support_pack=include_support_pack,
                append_response_format=append_response_format,
            )
        ]
    if dataset_format == EPISODE_GRPO_OFFLINE:
        return [_convert_offline_row(row, path=path)]
    if dataset_format == EPISODE_GRPO_ONLINE:
        return [_convert_online_row(row, path=path)]
    raise ValueError(f"Unsupported dataset format: {dataset_format}")


def _convert_source_episode_row(
    row: Mapping[str, Any],
    *,
    path: str,
    include_support_pack: bool,
    append_response_format: bool,
) -> dict[str, Any]:
    record = build_common_metadata(row, path)
    prompt = build_episode_prompt(
        row,
        include_support_pack=include_support_pack,
        append_response_format=append_response_format,
    )
    record.update(
        {
            "prompt": prompt,
            "main": dict(row["main"]),
            "probe": dict(row["probe"]),
            "support_pack": dict(row["support_pack"]),
            "reward_spec": dict(row.get("reward_spec", {})),
            "gold": {
                "main_gold_option": row["main"]["gold_option"],
                "main_gold_final_answer": row["main"]["gold_final_answer"],
                "probe_gold_option": row["probe"].get("gold_option"),
                "probe_gold_final_answer": row["probe"]["gold_final_answer"],
            },
            "dataset_format": SOURCE_EPISODE,
            "task_type": "episode",
        }
    )
    return record


def _convert_source_episode_split_rows(
    row: Mapping[str, Any],
    *,
    path: str,
    include_support_pack: bool,
    append_response_format: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task_type in ("main", "probe"):
        record = build_common_metadata(row, path)
        record.update(
            {
                "id": f"{row['episode_id']}::{task_type}",
                "episode_id": row["episode_id"],
                "task_type": task_type,
                "prompt": build_task_prompt(
                    row,
                    task_type=task_type,
                    include_support_pack=include_support_pack,
                    append_response_format=append_response_format,
                ),
                "main": dict(row["main"]),
                "probe": dict(row["probe"]),
                "support_pack": dict(row["support_pack"]),
                "reward_spec": dict(row.get("reward_spec", {})),
                "gold": {
                    "main_gold_option": row["main"]["gold_option"],
                    "main_gold_final_answer": row["main"]["gold_final_answer"],
                    "probe_gold_option": row["probe"].get("gold_option"),
                    "probe_gold_final_answer": row["probe"]["gold_final_answer"],
                },
                "dataset_format": SOURCE_EPISODE_SPLIT,
            }
        )
        rows.append(record)
    return rows


def _convert_offline_row(row: Mapping[str, Any], *, path: str) -> dict[str, Any]:
    record = build_common_metadata(row, path)
    responses = [dict(item) for item in row.get("responses", [])]
    best_response = max(responses, key=lambda item: float(item.get("score", 0.0)))
    targets = derive_episode_targets_from_text(row["prompt"], best_response["text"])
    record.update(
        {
            "prompt": str(row["prompt"]),
            "responses": responses,
            "candidate_count": len(responses),
            "candidate_scores": [float(item.get("score", 0.0)) for item in responses],
            "best_response_text": best_response["text"],
            "best_response_score": float(best_response.get("score", 0.0)),
            "reward_spec": dict(row.get("reward_spec", {})),
            "main": targets["main"],
            "probe": targets["probe"],
            "gold": {
                "main_gold_option": targets["main"]["gold_option"],
                "main_gold_final_answer": targets["main"]["gold_final_answer"],
                "probe_gold_final_answer": targets["probe"]["gold_final_answer"],
            },
            "dataset_format": EPISODE_GRPO_OFFLINE,
            "task_type": row.get("task_type", "episode"),
        }
    )
    return record


def _convert_online_row(row: Mapping[str, Any], *, path: str) -> dict[str, Any]:
    record = build_common_metadata(row, path)
    options = extract_prompt_options(str(row["prompt"]))
    gold = dict(row.get("gold", {}))
    task_type = row.get("task_type", "episode")
    record.update(
        {
            "prompt": str(row["prompt"]),
            "reward_spec": dict(row.get("reward_spec", {})),
            "main": {
                "options": [] if task_type == "probe" else options,
                "gold_option": gold.get("main_gold_option"),
                "gold_final_answer": gold.get("main_gold_final_answer"),
                "expected_intermediates": [],
                "operation_chain": [],
            },
            "probe": {
                "options": options if task_type == "probe" else gold.get("probe_options", []),
                "gold_option": gold.get("probe_gold_option"),
                "gold_final_answer": gold.get("probe_gold_final_answer"),
                "expected_intermediates": [],
                "operation_chain": [],
            },
            "gold": gold,
            "dataset_format": EPISODE_GRPO_ONLINE,
            "task_type": task_type,
        }
    )
    return record
