from __future__ import annotations

import importlib
import json
import random

from episode_builder import build_episode_from_seed
from mcq_consistency import validate_candidate_mcq_consistency, validate_episode_adversarial_candidates
from utils_episode import ensure_legacy_paths, load_json, resolve_repo_path

ensure_legacy_paths()

from schemas_seed import seed_from_dict  # type: ignore[import-not-found]


def test_probe_is_same_skill_and_different_from_main() -> None:
    seed_path = resolve_repo_path("seeds_llm_clean_A.jsonl")
    phrase_path = resolve_repo_path("phrase_bank_tierA_all.json")
    with seed_path.open("r", encoding="utf-8") as handle:
        row = next(line for line in handle if line.strip())
    seed = seed_from_dict(json.loads(row))
    phrase_bank = load_json(phrase_path)

    episode = build_episode_from_seed(seed, phrase_bank, random.Random(5), {"episode_id": "ep_test"})

    assert episode["main"]["gold_final_answer"] != episode["probe"]["gold_final_answer"]
    assert episode["probe"]["same_skill_as_main"] is True
    assert f"skill:{episode['skill_id']}" in episode["tags"]


def test_episode_builder_emits_mcq_consistent_adversarial_candidates() -> None:
    seed_path = resolve_repo_path("seeds_llm_clean_A.jsonl")
    phrase_path = resolve_repo_path("phrase_bank_tierA_all.json")
    with seed_path.open("r", encoding="utf-8") as handle:
        row = next(line for line in handle if line.strip())
    seed = seed_from_dict(json.loads(row))
    phrase_bank = load_json(phrase_path)

    episode = build_episode_from_seed(seed, phrase_bank, random.Random(5), {"episode_id": "ep_test_consistency"})

    validate_episode_adversarial_candidates(episode)
    for candidate in episode["adversarial_candidates"]:
        validate_candidate_mcq_consistency(
            candidate,
            episode["main"]["options"],
            episode["main"]["gold_option"],
            episode["main"]["gold_final_answer"],
            candidate["family"],
        )


def test_legacy_scripts_import_without_syntax_errors() -> None:
    assert importlib.import_module("generate_raw")
    assert importlib.import_module("convert")
