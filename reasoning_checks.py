"""Reward-facing consistency checks for episode candidates."""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, Sequence

from mcq_consistency import normalize_mcq_value, resolve_value_for_option


def _normalize_answer(value: Any) -> str:
    return normalize_mcq_value(value)


def final_answer_correct(predicted_final: Any, expected_final: Any) -> bool:
    """Check whether the final answer matches exactly."""
    return _normalize_answer(predicted_final) == _normalize_answer(expected_final)


def selected_option_correct(selected_option: str | None, gold_option: str) -> bool:
    """Check whether the chosen option label is correct."""
    return bool(selected_option) and selected_option == gold_option


def mentioned_intermediates_consistent(
    mentioned_intermediates: Iterable[Any],
    expected_intermediates: Sequence[dict[str, Any]],
) -> bool:
    """Ensure the candidate mentions the expected intermediate values in order."""
    expected_values = [str(item["value"]) for item in expected_intermediates]
    mentioned_values = [str(value) for value in mentioned_intermediates]
    if len(mentioned_values) < len(expected_values):
        return False
    return mentioned_values[: len(expected_values)] == expected_values


def operation_order_consistent(
    predicted_chain: Sequence[str],
    expected_chain: Sequence[str],
) -> bool:
    """Check whether the operation sequence matches the gold order."""
    return list(predicted_chain) == list(expected_chain)


def contradictions_detected(candidate: Dict[str, Any], main_item: Dict[str, Any]) -> bool:
    """Detect simple contradictions between option choice, final answer, and reasoning text."""
    selected_option = candidate.get("selected_option")
    final_answer = _normalize_answer(candidate.get("final_answer", ""))
    reasoning_text = str(candidate.get("reasoning_text", ""))

    if selected_option and resolve_value_for_option(main_item.get("options", []), selected_option) != final_answer:
        return True

    mentioned_numbers = re.findall(r"-?\d+", reasoning_text)
    if final_answer and mentioned_numbers and mentioned_numbers[-1] != final_answer:
        return True

    return False


def classify_correct_option_wrong_reasoning(candidate: Dict[str, Any], main_item: Dict[str, Any]) -> bool:
    """Flag candidates that select the correct option but provide inconsistent reasoning."""
    final_ok = final_answer_correct(candidate.get("final_answer"), main_item["gold_final_answer"])
    option_ok = selected_option_correct(candidate.get("selected_option"), main_item["gold_option"])
    intermediates_ok = mentioned_intermediates_consistent(
        candidate.get("mentioned_intermediates", []),
        main_item.get("expected_intermediates", []),
    )
    order_ok = operation_order_consistent(
        candidate.get("operation_chain", []),
        main_item.get("operation_chain", []),
    )
    contradiction = contradictions_detected(candidate, main_item)
    return final_ok and option_ok and (not intermediates_ok or not order_ok or contradiction)


def evaluate_main_response(candidate: Dict[str, Any], main_item: Dict[str, Any]) -> Dict[str, bool]:
    """Run the full reward-facing check set for a main answer."""
    checks = {
        "final_answer_correct": final_answer_correct(candidate.get("final_answer"), main_item["gold_final_answer"]),
        "selected_option_correct": selected_option_correct(candidate.get("selected_option"), main_item["gold_option"]),
        "mentioned_intermediates_consistent": mentioned_intermediates_consistent(
            candidate.get("mentioned_intermediates", []),
            main_item.get("expected_intermediates", []),
        ),
        "operation_order_consistent": operation_order_consistent(
            candidate.get("operation_chain", []),
            main_item.get("operation_chain", []),
        ),
        "contradictions_detected": contradictions_detected(candidate, main_item),
    }
    checks["correct_option_wrong_reasoning"] = classify_correct_option_wrong_reasoning(candidate, main_item)
    return checks
