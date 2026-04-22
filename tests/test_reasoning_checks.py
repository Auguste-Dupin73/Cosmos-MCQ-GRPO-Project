from __future__ import annotations

from reasoning_checks import (
    classify_correct_option_wrong_reasoning,
    evaluate_main_response,
    final_answer_correct,
)


def _main_item() -> dict:
    return {
        "gold_option": "B",
        "gold_final_answer": "42",
        "expected_intermediates": [{"name": "step1", "value": 21}, {"name": "step2", "value": 42}],
        "operation_chain": ["multiply", "add"],
        "options": [
            {"label": "A", "text": "41"},
            {"label": "B", "text": "42"},
            {"label": "C", "text": "43"},
            {"label": "D", "text": "21"},
        ],
    }


def test_reasoning_checker_detects_correct_answer() -> None:
    candidate = {
        "selected_option": "B",
        "final_answer": "42",
        "reasoning_text": "21 sonra 42 buldum. 42",
        "mentioned_intermediates": [21, 42],
        "operation_chain": ["multiply", "add"],
    }
    checks = evaluate_main_response(candidate, _main_item())
    assert final_answer_correct("42", "42")
    assert checks["final_answer_correct"] is True
    assert checks["correct_option_wrong_reasoning"] is False


def test_reasoning_checker_detects_wrong_final_answer() -> None:
    candidate = {
        "selected_option": "A",
        "final_answer": "41",
        "reasoning_text": "21 sonra 41 buldum. 41",
        "mentioned_intermediates": [21, 41],
        "operation_chain": ["multiply", "add"],
    }
    checks = evaluate_main_response(candidate, _main_item())
    assert checks["final_answer_correct"] is False


def test_reasoning_checker_detects_correct_option_wrong_reasoning() -> None:
    candidate = {
        "selected_option": "B",
        "final_answer": "42",
        "reasoning_text": "21 sonra 44 buldum. 42",
        "mentioned_intermediates": [21, 44],
        "operation_chain": ["add", "multiply"],
    }
    assert classify_correct_option_wrong_reasoning(candidate, _main_item()) is True
