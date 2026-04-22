"""Multiple-choice option generation for structured math items."""

from __future__ import annotations

import random
from typing import Any, Dict, List

from mcq_consistency import validate_mcq_options


OPTION_LABELS = ["A", "B", "C", "D"]


def _numeric_intermediates(structured_item: Dict[str, Any]) -> List[int]:
    values: list[int] = []
    for item in structured_item.get("expected_intermediates", []):
        try:
            values.append(int(item["value"]))
        except (KeyError, TypeError, ValueError):
            continue
    return values


def build_mcq(structured_item: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
    """Build four distinct options with exactly one gold answer."""
    gold_value = int(structured_item["gold_final_answer"])
    intermediates = [value for value in _numeric_intermediates(structured_item) if value != gold_value]

    candidates: list[tuple[str, int]] = []
    candidates.append(("near_miss_small_offset", gold_value + (1 if gold_value >= 0 else -1)))

    copied_operand = intermediates[0] if intermediates else gold_value + 3
    candidates.append(("copied_operand", copied_operand))

    omitted_step = intermediates[-1] if intermediates else gold_value + 7
    candidates.append(("omitted_step", omitted_step))

    if len(intermediates) >= 2:
        wrong_order = abs(intermediates[0] - intermediates[-1])
        if wrong_order == gold_value:
            wrong_order += 2
    else:
        wrong_order = gold_value + 9
    candidates.append(("wrong_operation_order", wrong_order))

    deduped: list[tuple[str, int]] = []
    seen = {gold_value}
    for family, value in candidates:
        if value in seen:
            continue
        seen.add(value)
        deduped.append((family, value))

    fallback_offset = 2
    while len(deduped) < 3:
        fallback = gold_value + fallback_offset
        fallback_offset += 2
        if fallback not in seen:
            seen.add(fallback)
            deduped.append(("fallback_numeric", fallback))

    option_payloads = [{"text": str(gold_value), "is_gold": True, "family": "gold"}]
    option_payloads.extend(
        {"text": str(value), "is_gold": False, "family": family}
        for family, value in deduped[:3]
    )

    rng.shuffle(option_payloads)
    options: list[dict[str, Any]] = []
    gold_label = ""
    for label, payload in zip(OPTION_LABELS, option_payloads):
        options.append({"label": label, "text": payload["text"], "family": payload["family"]})
        if payload["is_gold"]:
            gold_label = label

    validate_mcq_options(options)
    if not gold_label:
        raise ValueError("MCQ builder failed to assign a gold option")

    return {
        "mcq_stem": f"{structured_item['question_text']} Doğru seçeneği işaretleyin.",
        "options": options,
        "gold_option": gold_label,
    }
