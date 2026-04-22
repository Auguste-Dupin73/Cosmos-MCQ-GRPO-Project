"""Episode assembly for the new GRPO episode pipeline."""

from __future__ import annotations

import random
from typing import Any, Dict

from adversarial_negatives import build_adversarial_candidates
from mcq_builder import build_mcq
from mcq_consistency import normalize_episode_adversarial_candidates, validate_episode_adversarial_candidates
from render_structured import render_structured_seed
from skill_registry import get_skill_spec
from support_pack import build_support_pack


def _build_probe(seed: Any, phrase_bank: Dict[str, Any], rng: random.Random, main: Dict[str, Any]) -> Dict[str, Any]:
    for _ in range(24):
        probe = render_structured_seed(seed, phrase_bank, rng)
        if (
            probe["skill_id"] == main["skill_id"]
            and probe["gold_final_answer"] != main["gold_final_answer"]
            and probe["question_text"] != main["question_text"]
        ):
            probe["same_skill_as_main"] = True
            return probe
    raise ValueError(f"Could not build a numerically distinct probe for template: {seed.template}")


def build_episode_from_seed(
    seed: Any,
    phrase_bank: Dict[str, Any],
    rng: random.Random,
    config: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Build a full episode record from one seed definition."""
    config = config or {}
    main = render_structured_seed(seed, phrase_bank, rng)
    mcq = build_mcq(main, rng)
    main.update(mcq)

    probe = _build_probe(seed, phrase_bank, rng, main)
    spec = get_skill_spec(main["template_id"])
    support_pack = build_support_pack(main["skill_id"])

    episode_id = config.get("episode_id") or f"{seed.template}-{rng.randrange(1_000_000):06d}"
    tags = [
        f"skill:{main['skill_id']}",
        f"template:{main['template_id']}",
        f"tier:{main['tier']}",
        f"difficulty:{main['difficulty']}",
        f"topic:{spec.topic}",
        "lang:tr",
        "mcq:true",
        "probe:true",
        "reward:gated",
        f"operation_chain:{'+'.join(main['operation_chain'])}",
    ]

    episode = {
        "episode_id": episode_id,
        "skill_id": main["skill_id"],
        "template_id": main["template_id"],
        "tier": main["tier"],
        "difficulty": main["difficulty"],
        "language": "tr",
        "main": {
            "question_text": main["question_text"],
            "mcq_stem": main["mcq_stem"],
            "options": main["options"],
            "gold_option": main["gold_option"],
            "gold_final_answer": main["gold_final_answer"],
            "gold_solution_text": main["gold_solution_text"],
            "expected_intermediates": main["expected_intermediates"],
            "operation_chain": main["operation_chain"],
            "concrete_seed": main["concrete_seed"],
        },
        "probe": {
            "question_text": probe["question_text"],
            "gold_final_answer": probe["gold_final_answer"],
            "gold_solution_text": probe["gold_solution_text"],
            "expected_intermediates": probe["expected_intermediates"],
            "operation_chain": probe["operation_chain"],
            "same_skill_as_main": True,
            "concrete_seed": probe["concrete_seed"],
        },
        "support_pack": support_pack,
        "reward_spec": {
            "require_main_option_correct": True,
            "require_main_reasoning_consistent": True,
            "require_probe_correct": True,
            "allow_retry_with_support_pack": True,
            "scoring_mode": "gated",
        },
        "adversarial_candidates": [],
        "tags": tags,
    }
    episode["adversarial_candidates"] = build_adversarial_candidates(episode)
    episode["adversarial_candidates"] = normalize_episode_adversarial_candidates(episode)
    validate_episode_adversarial_candidates(episode)
    return episode
