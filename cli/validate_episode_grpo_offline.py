"""Validate MCQ consistency inside episode_grpo_offline.jsonl outputs."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

try:
    from cli._bootstrap import ensure_repo_root
except ImportError:  # pragma: no cover
    from _bootstrap import ensure_repo_root

ensure_repo_root()

from mcq_consistency import (
    MCQConsistencyError,
    extract_main_final_answer,
    extract_main_selected_option,
    resolve_option_for_value,
    validate_mcq_options,
)


OPTION_LINE_RE = re.compile(r"^([A-Z])\)\s*(.+?)\s*$")


def _prompt_options(prompt: str) -> list[dict[str, str]]:
    options = []
    for line in prompt.splitlines():
        match = OPTION_LINE_RE.match(line.strip())
        if match:
            options.append({"label": match.group(1), "text": match.group(2)})
    return options


def validate_offline_file(path: Path, max_samples: int = 5) -> dict[str, Any]:
    summary = {
        "total_candidates_scanned": 0,
        "mismatch_count": 0,
        "unmappable_final_answer_count": 0,
        "duplicate_option_count": 0,
        "samples": [],
    }

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            options = _prompt_options(row["prompt"])
            try:
                validate_mcq_options(options)
            except MCQConsistencyError as exc:
                summary["duplicate_option_count"] += 1
                if len(summary["samples"]) < max_samples:
                    summary["samples"].append(
                        {
                            "line": line_number,
                            "id": row.get("id"),
                            "error": str(exc),
                        }
                    )
                continue

            for response_index, response in enumerate(row.get("responses", [])):
                summary["total_candidates_scanned"] += 1
                selected_option = extract_main_selected_option(response["text"])
                final_answer = extract_main_final_answer(response["text"])
                if not selected_option or not final_answer:
                    summary["unmappable_final_answer_count"] += 1
                    if len(summary["samples"]) < max_samples:
                        summary["samples"].append(
                            {
                                "line": line_number,
                                "id": row.get("id"),
                                "response_index": response_index,
                                "error": "Missing selected option or final answer",
                            }
                        )
                    continue

                try:
                    resolved_option = resolve_option_for_value(options, final_answer)
                except MCQConsistencyError as exc:
                    summary["unmappable_final_answer_count"] += 1
                    if len(summary["samples"]) < max_samples:
                        summary["samples"].append(
                            {
                                "line": line_number,
                                "id": row.get("id"),
                                "response_index": response_index,
                                "error": str(exc),
                                "selected_option": selected_option,
                                "final_answer": final_answer,
                            }
                        )
                    continue

                if selected_option != resolved_option:
                    summary["mismatch_count"] += 1
                    if len(summary["samples"]) < max_samples:
                        summary["samples"].append(
                            {
                                "line": line_number,
                                "id": row.get("id"),
                                "response_index": response_index,
                                "selected_option": selected_option,
                                "resolved_option": resolved_option,
                                "final_answer": final_answer,
                            }
                        )

    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--max_samples", type=int, default=5)
    args = parser.parse_args()

    summary = validate_offline_file(Path(args.input), max_samples=args.max_samples)
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if (
        summary["mismatch_count"]
        or summary["unmappable_final_answer_count"]
        or summary["duplicate_option_count"]
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
