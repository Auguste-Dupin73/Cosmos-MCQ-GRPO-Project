"""Validate MCQ consistency inside source episode JSONL files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from cli._bootstrap import ensure_repo_root
except ImportError:  # pragma: no cover
    from _bootstrap import ensure_repo_root

ensure_repo_root()

from mcq_consistency import MCQConsistencyError, validate_episode_adversarial_candidates, validate_mcq_options


def validate_episode_file(path: Path, max_samples: int = 5) -> dict[str, Any]:
    summary = {
        "total_episodes": 0,
        "total_adversarial_candidates": 0,
        "mismatch_count": 0,
        "unmappable_candidate_count": 0,
        "duplicate_option_count": 0,
        "family_failure_breakdown": {},
        "samples": [],
    }

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            episode = json.loads(line)
            summary["total_episodes"] += 1
            summary["total_adversarial_candidates"] += len(episode.get("adversarial_candidates", []))

            try:
                validate_mcq_options(episode["main"]["options"])
            except MCQConsistencyError as exc:
                summary["duplicate_option_count"] += 1
                if len(summary["samples"]) < max_samples:
                    summary["samples"].append(
                        {
                            "line": line_number,
                            "episode_id": episode.get("episode_id"),
                            "error": str(exc),
                        }
                    )
                continue

            for candidate in episode.get("adversarial_candidates", []):
                try:
                    validate_episode_adversarial_candidates(
                        {
                            "main": episode["main"],
                            "adversarial_candidates": [candidate],
                        }
                    )
                except MCQConsistencyError as exc:
                    family = str(candidate.get("family", "unknown"))
                    breakdown = summary["family_failure_breakdown"]
                    breakdown[family] = breakdown.get(family, 0) + 1
                    error_text = str(exc)
                    if "unique option for value" in error_text or "missing a final answer" in error_text:
                        summary["unmappable_candidate_count"] += 1
                    else:
                        summary["mismatch_count"] += 1
                    if len(summary["samples"]) < max_samples:
                        summary["samples"].append(
                            {
                                "line": line_number,
                                "episode_id": episode.get("episode_id"),
                                "candidate_id": candidate.get("candidate_id"),
                                "family": family,
                                "error": error_text,
                            }
                        )

    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--max_samples", type=int, default=5)
    args = parser.parse_args()

    summary = validate_episode_file(Path(args.input), max_samples=args.max_samples)
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if (
        summary["mismatch_count"]
        or summary["unmappable_candidate_count"]
        or summary["duplicate_option_count"]
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
