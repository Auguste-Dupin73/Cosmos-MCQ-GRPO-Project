from __future__ import annotations

import random

from mcq_builder import build_mcq


def test_mcq_has_exactly_one_gold_option() -> None:
    structured = {
        "question_text": "Örnek soru",
        "gold_final_answer": "42",
        "expected_intermediates": [{"name": "step1", "value": 21}, {"name": "step2", "value": 42}],
        "operation_chain": ["multiply"],
    }
    mcq = build_mcq(structured, random.Random(7))

    assert len(mcq["options"]) == 4
    assert len({option["text"] for option in mcq["options"]}) == 4
    assert sum(1 for option in mcq["options"] if option["label"] == mcq["gold_option"] and option["text"] == "42") == 1
