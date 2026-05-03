from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cli.convert_episodes import convert_episode_rows
from episode_builder import build_episode_from_seed
from mcq_consistency import validate_episode_adversarial_candidates
from utils_episode import ensure_legacy_paths, load_json, load_jsonl, resolve_repo_path, write_jsonl

ensure_legacy_paths()

from schemas_seed import seed_from_dict  # type: ignore[import-not-found]

try:
    from validate_seeds import validate_seed  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover
    validate_seed = None


DEFAULT_TIER_COUNTS = {"A": 250, "B": 150, "C": 100}
DEFAULT_PHRASE_BANKS = {
    "A": "example-generator-v1/data/phrase_bank_tierA_all.json",
    "B": "example-generator-v1/data/phrase_bank_tierB_all.json",
    "C": "example-generator-v1/data/phrase_bank_tierC_all.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a tier-balanced source episode dataset.")
    parser.add_argument("--out_dir", required=True, help="Output directory for episodes, selected seeds, and manifest.")
    parser.add_argument("--seeds_root", default="example-generator-v1/data/team_seeds")
    parser.add_argument("--tier_count", action="append", default=None, help="Tier count as A=250. Repeatable.")
    parser.add_argument("--variants_per_seed", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260503)
    parser.add_argument("--id_prefix", default="test_500")
    parser.add_argument("--exclude_selected_dir", action="append", default=None)
    parser.add_argument("--exclude_episodes", action="append", default=None)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--no_convert", action="store_true", help="Only write source episodes and manifest.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tier_counts = parse_tier_counts(args.tier_count)
    excluded_seed_keys = load_excluded_seed_keys(args.exclude_selected_dir or [])
    excluded_question_keys = load_excluded_question_keys(args.exclude_episodes or [])
    initial_excluded_question_count = len(excluded_question_keys)
    used_question_keys = set(excluded_question_keys)

    rng = random.Random(args.seed)
    all_episodes: list[dict[str, Any]] = []
    selected_seed_files: dict[str, Any] = {}
    episode_files: dict[str, Any] = {}
    skipped: dict[str, Counter[str]] = {}

    for tier, target_count in tier_counts.items():
        tier_rng = random.Random(args.seed + ord(tier))
        phrase_bank = load_json(resolve_repo_path(DEFAULT_PHRASE_BANKS[tier]))
        seed_rows = collect_seed_rows(resolve_repo_path(str(Path(args.seeds_root) / f"tier{tier}")))
        tier_episodes, selected_rows, skipped_counter = select_buildable_episodes(
            seed_rows=seed_rows,
            tier=tier,
            target_count=target_count,
            variants_per_seed=args.variants_per_seed,
            phrase_bank=phrase_bank,
            rng=tier_rng,
            id_prefix=args.id_prefix,
            excluded_seed_keys=excluded_seed_keys,
            used_question_keys=used_question_keys,
        )
        skipped[tier] = skipped_counter

        selected_path = out_dir / "selected_seeds" / f"tier{tier}_selected_{len(selected_rows)}.jsonl"
        tier_episode_path = out_dir / f"tier{tier}_episodes.jsonl"
        write_jsonl(selected_path, selected_rows)
        write_jsonl(tier_episode_path, tier_episodes)
        selected_seed_files[tier] = {
            "path": str(selected_path),
            "count": len(selected_rows),
            "template_counts": dict(Counter(row.get("template") for row in selected_rows)),
        }
        episode_files[tier] = {"path": str(tier_episode_path), "count": len(tier_episodes)}
        all_episodes.extend(tier_episodes)

    if args.shuffle:
        rng.shuffle(all_episodes)

    episodes_path = out_dir / "episodes.jsonl"
    write_jsonl(episodes_path, all_episodes)
    if not args.no_convert:
        convert_episode_rows(all_episodes, out_dir)

    manifest = {
        "target_total_episodes": sum(tier_counts.values()) * args.variants_per_seed,
        "actual_total_episodes": len(all_episodes),
        "variants_per_seed": args.variants_per_seed,
        "tier_plan": {
            tier: {"seed_count": count, "episode_count": count * args.variants_per_seed}
            for tier, count in tier_counts.items()
        },
        "sampling_seed": args.seed,
        "id_prefix": args.id_prefix,
        "excluded_seed_count": len(excluded_seed_keys),
        "excluded_question_count": initial_excluded_question_count,
        "generated_question_count": len(used_question_keys) - initial_excluded_question_count,
        "selected_seed_files": selected_seed_files,
        "episode_files": episode_files,
        "skipped_candidate_counts": {tier: dict(counter) for tier, counter in skipped.items()},
        "combined_episode_file": {"path": str(episodes_path), "count": len(all_episodes)},
        "converted": not args.no_convert,
        "shuffled": bool(args.shuffle),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def parse_tier_counts(raw_values: list[str] | None) -> dict[str, int]:
    if not raw_values:
        return dict(DEFAULT_TIER_COUNTS)
    counts: dict[str, int] = {}
    for raw_value in raw_values:
        tier, value = raw_value.split("=", 1)
        tier = tier.strip().upper()
        if tier not in DEFAULT_TIER_COUNTS:
            raise ValueError(f"Unsupported tier: {tier}")
        counts[tier] = int(value)
    return counts


def collect_seed_rows(tier_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in sorted(tier_dir.glob("*.jsonl")):
        for row in load_jsonl(path):
            key = canonical_seed_key(row)
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)
    return rows


def select_buildable_episodes(
    *,
    seed_rows: list[dict[str, Any]],
    tier: str,
    target_count: int,
    variants_per_seed: int,
    phrase_bank: dict[str, Any],
    rng: random.Random,
    id_prefix: str,
    excluded_seed_keys: set[str],
    used_question_keys: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Counter[str]]:
    candidates = list(seed_rows)
    rng.shuffle(candidates)
    selected_rows: list[dict[str, Any]] = []
    episodes: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()

    for row in candidates:
        if len(selected_rows) >= target_count:
            break
        if canonical_seed_key(row) in excluded_seed_keys:
            skipped["excluded_training_seed"] += 1
            continue
        try:
            seed = seed_from_dict(row)
            if validate_seed is not None:
                validate_seed(seed)
            built = []
            for variant_index in range(variants_per_seed):
                episode_id = f"{id_prefix}_{tier}_{len(selected_rows) + 1:03d}_{seed.template}_{variant_index + 1}"
                episode = build_episode_from_seed(
                    seed=seed,
                    phrase_bank=phrase_bank,
                    rng=rng,
                    config={"episode_id": episode_id},
                )
                validate_episode_adversarial_candidates(episode)
                question_keys = {
                    question_key(episode["main"]["mcq_stem"]),
                    question_key(episode["probe"]["question_text"]),
                }
                if used_question_keys.intersection(question_keys):
                    raise DuplicateQuestionError
                built.append((episode, question_keys))
        except DuplicateQuestionError:
            skipped["duplicate_question"] += 1
            continue
        except Exception as exc:  # keep sampling robust across heterogeneous seed files
            skipped[f"build_error:{type(exc).__name__}"] += 1
            continue

        selected_rows.append(row)
        for episode, question_keys in built:
            episodes.append(episode)
            used_question_keys.update(question_keys)

    if len(selected_rows) < target_count:
        raise RuntimeError(f"Tier {tier} only produced {len(selected_rows)} buildable seeds; target was {target_count}")
    return episodes, selected_rows, skipped


def load_excluded_seed_keys(paths: Iterable[str]) -> set[str]:
    excluded: set[str] = set()
    for raw_path in paths:
        path = resolve_repo_path(raw_path)
        files = sorted(path.glob("*.jsonl")) if path.is_dir() else [path]
        for file_path in files:
            for row in load_jsonl(file_path):
                excluded.add(canonical_seed_key(row))
    return excluded


def load_excluded_question_keys(paths: Iterable[str]) -> set[str]:
    excluded: set[str] = set()
    for raw_path in paths:
        for episode in load_jsonl(resolve_repo_path(raw_path)):
            excluded.add(question_key(episode["main"]["mcq_stem"]))
            excluded.add(question_key(episode["probe"]["question_text"]))
    return excluded


def canonical_seed_key(row: dict[str, Any]) -> str:
    return json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def question_key(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


class DuplicateQuestionError(Exception):
    pass


if __name__ == "__main__":
    main()
