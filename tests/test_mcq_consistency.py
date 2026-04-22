from __future__ import annotations

import json
import random
import re
from pathlib import Path

import pytest

from cli.convert_episodes import convert_episode_rows
from mcq_builder import build_mcq
from mcq_consistency import (
    MCQConsistencyError,
    extract_main_final_answer,
    extract_main_selected_option,
    normalize_candidate_against_mcq,
    resolve_option_for_value,
    validate_candidate_mcq_consistency,
    validate_mcq_options,
)


OPTION_LINE_RE = re.compile(r"^([A-Z])\)\s*(.+?)\s*$")


def _episode() -> dict:
    return {
        "episode_id": "ep_mcq_consistency",
        "skill_id": "net_remainder_revenue",
        "template_id": "sell_leftover",
        "tier": "A",
        "difficulty": "easy",
        "language": "tr",
        "main": {
            "question_text": "Elif her gun 24 tane yumurta bulur.",
            "mcq_stem": "Elif her gun 24 tane yumurta bulur. Dogru secenegi isaretleyin.",
            "options": [
                {"label": "A", "text": "56", "family": "gold"},
                {"label": "B", "text": "14", "family": "copied_operand"},
                {"label": "C", "text": "65", "family": "wrong_operation_order"},
                {"label": "D", "text": "57", "family": "near_miss_small_offset"},
            ],
            "gold_option": "A",
            "gold_final_answer": "56",
            "gold_solution_text": "Once kalan miktari buluruz: 24 - 4 - 6 = 14. Sonra satis gelirini hesaplariz: 14 x 4 = 56.",
            "expected_intermediates": [
                {"name": "remaining_after_use", "value": 14},
                {"name": "sales_total", "value": 56},
            ],
            "operation_chain": ["subtract", "subtract", "multiply"],
            "concrete_seed": {"template": "sell_leftover"},
        },
        "probe": {
            "question_text": "Probe soru",
            "gold_final_answer": "36",
            "gold_solution_text": "Probe cozum metni.",
            "expected_intermediates": [
                {"name": "remaining_after_use", "value": 9},
                {"name": "sales_total", "value": 36},
            ],
            "operation_chain": ["subtract", "subtract", "multiply"],
            "same_skill_as_main": True,
            "concrete_seed": {"template": "sell_leftover"},
        },
        "support_pack": {
            "skill_summary": "Once kalan miktari bul, sonra birim fiyatla carp.",
            "formula_hints": ["kalan = toplam - kullanim1 - kullanim2"],
            "mini_example": "Mini ornek",
        },
        "reward_spec": {
            "require_main_option_correct": True,
            "require_main_reasoning_consistent": True,
            "require_probe_correct": True,
            "allow_retry_with_support_pack": True,
            "scoring_mode": "gated",
        },
        "tags": ["skill:net_remainder_revenue"],
        "adversarial_candidates": [
            {
                "candidate_id": "ep_mcq_consistency::wrong_final",
                "family": "wrong_final",
                "main_response": {
                    "selected_option": "C",
                    "final_answer": "14",
                    "reasoning_text": "Sonucu yanlış işlemle 14 buldum.",
                    "mentioned_intermediates": [14],
                    "operation_chain": ["subtract", "subtract", "multiply"],
                },
                "probe_response": None,
                "text": "Ana soru seçimi: C\nAna çözüm: Sonucu yanlış işlemle 14 buldum.\nAna nihai cevap: 14",
            },
            {
                "candidate_id": "ep_mcq_consistency::near_miss",
                "family": "near_miss",
                "main_response": {
                    "selected_option": "B",
                    "final_answer": "57",
                    "reasoning_text": "Neredeyse doğru gidip sonucu 57 yazdım.",
                    "mentioned_intermediates": [14, 57],
                    "operation_chain": ["subtract", "subtract", "multiply"],
                },
                "probe_response": None,
                "text": "Ana soru seçimi: B\nAna çözüm: Neredeyse doğru gidip sonucu 57 yazdım.\nAna nihai cevap: 57",
            },
            {
                "candidate_id": "ep_mcq_consistency::copied_operand",
                "family": "copied_operand",
                "main_response": {
                    "selected_option": "B",
                    "final_answer": "999",
                    "reasoning_text": "Ara değeri son cevap sanıp doğrudan yazdım.",
                    "mentioned_intermediates": [14, 56],
                    "operation_chain": ["subtract", "subtract", "multiply"],
                },
                "probe_response": None,
                "text": "Ana soru seçimi: B\nAna çözüm: Ara değeri son cevap sanıp doğrudan yazdım.\nAna nihai cevap: 999",
            },
            {
                "candidate_id": "ep_mcq_consistency::wrong_operation_order",
                "family": "wrong_operation_order",
                "main_response": {
                    "selected_option": "B",
                    "final_answer": "65",
                    "reasoning_text": "İşlemleri ters sırayla uyguladım.",
                    "mentioned_intermediates": [56, 14],
                    "operation_chain": ["multiply", "subtract", "subtract"],
                },
                "probe_response": None,
                "text": "Ana soru seçimi: B\nAna çözüm: İşlemleri ters sırayla uyguladım.\nAna nihai cevap: 65",
            },
            {
                "candidate_id": "ep_mcq_consistency::correct_option_wrong_reasoning",
                "family": "correct_option_wrong_reasoning",
                "main_response": {
                    "selected_option": "D",
                    "final_answer": "57",
                    "reasoning_text": "Doğru seçeneği işaretledim ama ara işlem sırasını karıştırdım.",
                    "mentioned_intermediates": [14, 58],
                    "operation_chain": ["multiply", "subtract", "subtract"],
                },
                "probe_response": None,
                "text": "Ana soru seçimi: D\nAna çözüm: Doğru seçeneği işaretledim ama ara işlem sırasını karıştırdım.\nAna nihai cevap: 57",
            },
            {
                "candidate_id": "ep_mcq_consistency::correct_main_wrong_probe",
                "family": "correct_main_wrong_probe",
                "main_response": {
                    "selected_option": "A",
                    "final_answer": "56",
                    "reasoning_text": "Once kalan miktari buluruz: 24 - 4 - 6 = 14. Sonra satis gelirini hesaplariz: 14 x 4 = 56.",
                    "mentioned_intermediates": [14, 56],
                    "operation_chain": ["subtract", "subtract", "multiply"],
                },
                "probe_response": {
                    "final_answer": "37",
                    "reasoning_text": "Probe sorusunda son adimda hata yaptim.",
                    "mentioned_intermediates": [9, 37],
                    "operation_chain": ["subtract", "subtract", "multiply"],
                },
                "text": "Ana soru seçimi: A\nAna çözüm: Once kalan miktari buluruz: 24 - 4 - 6 = 14. Sonra satis gelirini hesaplariz: 14 x 4 = 56.\nAna nihai cevap: 56\nProbe çözüm: Probe sorusunda son adimda hata yaptim.\nProbe nihai cevap: 37",
            },
        ],
    }


def _normalize(candidate: dict) -> dict:
    episode = _episode()
    main = episode["main"]
    return normalize_candidate_against_mcq(
        candidate,
        main["options"],
        main["gold_option"],
        main["gold_final_answer"],
        candidate["family"],
        main_item=main,
    )


def _prompt_options(prompt: str) -> list[dict[str, str]]:
    options = []
    for line in prompt.splitlines():
        match = OPTION_LINE_RE.match(line.strip())
        if match:
            options.append({"label": match.group(1), "text": match.group(2)})
    return options


def test_wrong_final_candidate_has_selected_option_matching_wrong_final_answer() -> None:
    candidate = _episode()["adversarial_candidates"][0]
    normalized = _normalize(candidate)

    assert normalized["main_response"]["final_answer"] == "14"
    assert normalized["main_response"]["selected_option"] == "B"


def test_near_miss_candidate_has_selected_option_matching_near_miss_final_answer() -> None:
    candidate = _episode()["adversarial_candidates"][1]
    normalized = _normalize(candidate)

    assert normalized["main_response"]["final_answer"] == "57"
    assert normalized["main_response"]["selected_option"] == "D"


def test_correct_option_wrong_reasoning_keeps_gold_option_and_gold_final_answer() -> None:
    candidate = _episode()["adversarial_candidates"][4]
    normalized = _normalize(candidate)

    assert normalized["main_response"]["selected_option"] == "A"
    assert normalized["main_response"]["final_answer"] == "56"
    assert normalized["main_response"]["reasoning_text"] == candidate["main_response"]["reasoning_text"]


def test_impossible_final_answer_is_rejected_then_repaired() -> None:
    candidate = _episode()["adversarial_candidates"][2]
    main = _episode()["main"]

    with pytest.raises(MCQConsistencyError):
        validate_candidate_mcq_consistency(
            candidate,
            main["options"],
            main["gold_option"],
            main["gold_final_answer"],
            candidate["family"],
        )

    normalized = _normalize(candidate)
    assert normalized["main_response"]["final_answer"] == "14"
    assert normalized["main_response"]["selected_option"] == "B"


def test_convert_episode_rows_emits_mcq_consistent_offline_responses() -> None:
    episode = _episode()
    out_dir = Path("tmp_test_runs") / "mcq_consistency_convert"
    out_dir.mkdir(parents=True, exist_ok=True)
    convert_episode_rows([episode], out_dir)

    offline_path = out_dir / "episode_grpo_offline.jsonl"
    rows = [json.loads(line) for line in offline_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 1

    row = rows[0]
    options = _prompt_options(row["prompt"])
    validate_mcq_options(options)

    for response in row["responses"]:
        selected_option = extract_main_selected_option(response["text"])
        final_answer = extract_main_final_answer(response["text"])
        assert selected_option is not None
        assert final_answer is not None
        assert selected_option == resolve_option_for_value(options, final_answer)


def test_mcq_options_remain_unique_and_resolvable() -> None:
    structured = {
        "question_text": "Ornek soru",
        "gold_final_answer": "42",
        "expected_intermediates": [{"name": "step1", "value": 21}, {"name": "step2", "value": 42}],
        "operation_chain": ["multiply"],
    }
    mcq = build_mcq(structured, random.Random(7))
    validate_mcq_options(mcq["options"])

    for option in mcq["options"]:
        assert resolve_option_for_value(mcq["options"], option["text"]) == option["label"]

    with pytest.raises(MCQConsistencyError):
        validate_mcq_options(
            [
                {"label": "A", "text": "10"},
                {"label": "B", "text": "10"},
                {"label": "C", "text": "12"},
                {"label": "D", "text": "13"},
            ]
        )
