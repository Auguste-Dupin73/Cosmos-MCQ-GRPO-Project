"""Adversarial response generation for episode-style GRPO supervision."""

from __future__ import annotations

from typing import Any, Dict, List

from mcq_consistency import (
    choose_preferred_wrong_option,
    compose_episode_candidate_text,
    normalize_candidate_against_mcq,
    resolve_value_for_option,
)


def _candidate_label_and_value(main_item: Dict[str, Any], candidate_type: str) -> tuple[str, str]:
    label = choose_preferred_wrong_option(main_item, candidate_type)
    return label, resolve_value_for_option(main_item["options"], label)


def build_gold_episode_response(episode: Dict[str, Any]) -> Dict[str, Any]:
    """Build the ideal full-episode response."""
    main = episode["main"]
    probe = episode["probe"]
    candidate = {
        "candidate_id": f"{episode['episode_id']}::gold",
        "family": "gold",
        "main_response": {
            "selected_option": main["gold_option"],
            "final_answer": main["gold_final_answer"],
            "reasoning_text": main["gold_solution_text"],
            "mentioned_intermediates": [item["value"] for item in main["expected_intermediates"]],
            "operation_chain": list(main["operation_chain"]),
        },
        "probe_response": {
            "final_answer": probe["gold_final_answer"],
            "reasoning_text": probe["gold_solution_text"],
            "mentioned_intermediates": [item["value"] for item in probe["expected_intermediates"]],
            "operation_chain": list(probe["operation_chain"]),
        },
    }
    return normalize_candidate_against_mcq(
        candidate,
        main["options"],
        main["gold_option"],
        main["gold_final_answer"],
        "gold",
        main_item=main,
    )


def build_adversarial_candidates(episode: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate fixed adversarial families for an episode."""
    main = episode["main"]
    probe = episode["probe"]
    gold_intermediates = [item["value"] for item in main["expected_intermediates"]]
    probe_intermediates = [item["value"] for item in probe["expected_intermediates"]]
    mutated_intermediates = list(gold_intermediates)
    if mutated_intermediates:
        mutated_intermediates[-1] = int(main["gold_final_answer"]) + 2

    wrong_final_option, wrong_final_value = _candidate_label_and_value(main, "wrong_final")
    near_miss_option, near_miss_value = _candidate_label_and_value(main, "near_miss")
    copied_operand_option, copied_operand_value = _candidate_label_and_value(main, "copied_operand")
    wrong_order_option, wrong_order_value = _candidate_label_and_value(main, "wrong_operation_order")

    candidates = [
        {
            "candidate_id": f"{episode['episode_id']}::wrong_final",
            "family": "wrong_final",
            "main_response": {
                "selected_option": wrong_final_option,
                "final_answer": wrong_final_value,
                "reasoning_text": f"Sonucu yanlış işlemle {wrong_final_value} buldum.",
                "mentioned_intermediates": gold_intermediates[:-1] + [wrong_final_value],
                "operation_chain": list(main["operation_chain"]),
            },
            "probe_response": None,
        },
        {
            "candidate_id": f"{episode['episode_id']}::near_miss",
            "family": "near_miss",
            "main_response": {
                "selected_option": near_miss_option,
                "final_answer": near_miss_value,
                "reasoning_text": f"Neredeyse doğru gidip sonucu {near_miss_value} yazdım.",
                "mentioned_intermediates": gold_intermediates[:-1] + [near_miss_value],
                "operation_chain": list(main["operation_chain"]),
            },
            "probe_response": None,
        },
        {
            "candidate_id": f"{episode['episode_id']}::copied_operand",
            "family": "copied_operand",
            "main_response": {
                "selected_option": copied_operand_option,
                "final_answer": copied_operand_value,
                "reasoning_text": "Ara değeri son cevap sanıp doğrudan yazdım.",
                "mentioned_intermediates": gold_intermediates,
                "operation_chain": list(main["operation_chain"]),
            },
            "probe_response": None,
        },
        {
            "candidate_id": f"{episode['episode_id']}::wrong_operation_order",
            "family": "wrong_operation_order",
            "main_response": {
                "selected_option": wrong_order_option,
                "final_answer": wrong_order_value,
                "reasoning_text": "İşlemleri ters sırayla uyguladım.",
                "mentioned_intermediates": list(reversed(gold_intermediates)),
                "operation_chain": list(reversed(main["operation_chain"])),
            },
            "probe_response": None,
        },
        {
            "candidate_id": f"{episode['episode_id']}::correct_option_wrong_reasoning",
            "family": "correct_option_wrong_reasoning",
            "main_response": {
                "selected_option": main["gold_option"],
                "final_answer": main["gold_final_answer"],
                "reasoning_text": "Doğru seçeneği işaretledim ama ara işlem sırasını karıştırdım.",
                "mentioned_intermediates": mutated_intermediates,
                "operation_chain": list(reversed(main["operation_chain"])),
            },
            "probe_response": None,
        },
        {
            "candidate_id": f"{episode['episode_id']}::correct_main_wrong_probe",
            "family": "correct_main_wrong_probe",
            "main_response": {
                "selected_option": main["gold_option"],
                "final_answer": main["gold_final_answer"],
                "reasoning_text": main["gold_solution_text"],
                "mentioned_intermediates": gold_intermediates,
                "operation_chain": list(main["operation_chain"]),
            },
            "probe_response": {
                "final_answer": str(int(probe["gold_final_answer"]) + 1),
                "reasoning_text": "Probe sorusunda son adımda hata yaptım.",
                "mentioned_intermediates": probe_intermediates[:-1] + [int(probe["gold_final_answer"]) + 1],
                "operation_chain": list(probe["operation_chain"]),
            },
        },
    ]

    normalized_candidates: list[Dict[str, Any]] = []
    for candidate in candidates:
        candidate["text"] = compose_episode_candidate_text(candidate["main_response"], candidate.get("probe_response"))
        normalized_candidates.append(
            normalize_candidate_against_mcq(
                candidate,
                main["options"],
                main["gold_option"],
                main["gold_final_answer"],
                candidate["family"],
                main_item=main,
            )
        )
    return normalized_candidates
