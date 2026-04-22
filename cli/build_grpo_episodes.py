"""CLI for direct seed-to-episode generation."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

try:
    from cli._bootstrap import ensure_repo_root
except ImportError:  # pragma: no cover
    from _bootstrap import ensure_repo_root

ensure_repo_root()

from episode_builder import build_episode_from_seed
from mcq_consistency import validate_episode_adversarial_candidates
from utils_episode import ensure_legacy_paths, load_json, resolve_repo_path, write_jsonl

ensure_legacy_paths()

from schemas_seed import seed_from_dict  # type: ignore[import-not-found]

try:
    from validate_seeds import validate_seed  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover
    validate_seed = None


def _load_seed_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--phrase_bank", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--variants_per_seed", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    seed_path = resolve_repo_path(args.seeds)
    phrase_bank_path = resolve_repo_path(args.phrase_bank)
    out_path = Path(args.out)

    seed_rows = _load_seed_rows(seed_path)
    phrase_bank = load_json(phrase_bank_path)
    rng = random.Random(args.seed)

    episodes = []
    counter = 0
    for row in seed_rows:
        seed = seed_from_dict(row)
        if validate_seed is not None:
            validate_seed(seed)
        for variant_idx in range(args.variants_per_seed):
            counter += 1
            episode_id = f"ep_{counter:06d}_{seed.template}_{variant_idx + 1}"
            episode = build_episode_from_seed(
                seed=seed,
                phrase_bank=phrase_bank,
                rng=rng,
                config={"episode_id": episode_id},
            )
            validate_episode_adversarial_candidates(episode)
            episodes.append(episode)

    write_jsonl(out_path, episodes)
    print(f"Wrote {len(episodes)} episodes to {out_path}")


if __name__ == "__main__":
    main()
