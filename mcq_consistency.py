"""Central MCQ normalization and validation for episode candidates."""

from __future__ import annotations

import copy
import re
from typing import Any, Dict, Mapping, Sequence


MAIN_SELECTED_OPTION_RE = re.compile(r"^Ana soru seçimi:\s*([A-Z])\s*$", re.MULTILINE)
MAIN_FINAL_ANSWER_RE = re.compile(r"^Ana nihai cevap:\s*(.+?)\s*$", re.MULTILINE)

FULLY_CORRECT_FAMILIES = {"gold", "fully_correct"}
CORRECT_MAIN_FAMILIES = FULLY_CORRECT_FAMILIES | {
    "correct_main_wrong_probe",
    "correct_option_wrong_reasoning",
}
WRONG_MAIN_FAMILIES = {
    "copied_operand",
    "near_miss",
    "wrong_final",
    "wrong_operation_order",
}


class MCQConsistencyError(ValueError):
    """Raised when a candidate cannot be aligned to a valid MCQ mapping."""


def normalize_mcq_value(value: Any) -> str:
    """Normalize numeric-looking MCQ values into the stored string form."""
    return str(value).strip()


def validate_mcq_options(options: Sequence[Mapping[str, Any]]) -> None:
    """Ensure every option label and value resolves unambiguously."""
    label_to_value: dict[str, str] = {}
    value_to_label: dict[str, str] = {}
    for option in options:
        label = str(option.get("label", "")).strip()
        value = normalize_mcq_value(option.get("text", ""))
        if not label:
            raise MCQConsistencyError("MCQ option is missing a label")
        if not value:
            raise MCQConsistencyError(f"MCQ option {label} is missing a value")
        if label in label_to_value:
            raise MCQConsistencyError(f"Duplicate MCQ option label: {label}")
        if value in value_to_label:
            raise MCQConsistencyError(
                f"Duplicate MCQ option value {value!r} for labels {value_to_label[value]} and {label}"
            )
        label_to_value[label] = value
        value_to_label[value] = label


def resolve_option_for_value(options: Sequence[Mapping[str, Any]], value: Any) -> str:
    """Return the unique MCQ label that maps to a value."""
    validate_mcq_options(options)
    normalized_value = normalize_mcq_value(value)
    matches = [str(option["label"]).strip() for option in options if normalize_mcq_value(option["text"]) == normalized_value]
    if len(matches) != 1:
        raise MCQConsistencyError(f"Could not resolve a unique option for value {normalized_value!r}")
    return matches[0]


def resolve_value_for_option(options: Sequence[Mapping[str, Any]], label: str) -> str:
    """Return the unique value attached to one MCQ label."""
    validate_mcq_options(options)
    normalized_label = str(label).strip()
    matches = [normalize_mcq_value(option["text"]) for option in options if str(option["label"]).strip() == normalized_label]
    if len(matches) != 1:
        raise MCQConsistencyError(f"Could not resolve a unique value for option {normalized_label!r}")
    return matches[0]


def extract_main_selected_option(text: str) -> str | None:
    """Extract the selected main MCQ option from serialized candidate text."""
    match = MAIN_SELECTED_OPTION_RE.search(text)
    if not match:
        return None
    return match.group(1).strip()


def extract_main_final_answer(text: str) -> str | None:
    """Extract the selected main final answer from serialized candidate text."""
    match = MAIN_FINAL_ANSWER_RE.search(text)
    if not match:
        return None
    return match.group(1).strip()


def compose_episode_candidate_text(
    main_response: Mapping[str, Any],
    probe_response: Mapping[str, Any] | None = None,
) -> str:
    """Render the canonical serialized text for one episode candidate."""
    lines = [
        f"Ana soru seçimi: {main_response['selected_option']}",
        f"Ana çözüm: {main_response['reasoning_text']}",
        f"Ana nihai cevap: {main_response['final_answer']}",
    ]
    if probe_response is not None:
        lines.extend(
            [
                f"Probe çözüm: {probe_response['reasoning_text']}",
                f"Probe nihai cevap: {probe_response['final_answer']}",
            ]
        )
    return "\n".join(lines)


def choose_preferred_wrong_option(
    main_item: Mapping[str, Any],
    candidate_type: str,
    selected_option: str | None = None,
) -> str:
    """Pick a valid non-gold option for one wrong-answer candidate family."""
    options = main_item["options"]
    gold_option = str(main_item["gold_option"]).strip()
    validate_mcq_options(options)

    if selected_option:
        normalized_selected = str(selected_option).strip()
    else:
        normalized_selected = ""

    def find_family_option(family: str) -> str | None:
        for option in options:
            label = str(option["label"]).strip()
            if label == gold_option:
                continue
            if option.get("family") == family:
                return label
        return None

    if candidate_type == "near_miss":
        label = find_family_option("near_miss_small_offset")
        if label:
            return label
        try:
            gold_value = int(normalize_mcq_value(main_item["gold_final_answer"]))
        except (TypeError, ValueError):
            gold_value = None
        if gold_value is not None:
            numeric_candidates: list[tuple[int, str]] = []
            for option in options:
                label = str(option["label"]).strip()
                if label == gold_option:
                    continue
                try:
                    distance = abs(int(normalize_mcq_value(option["text"])) - gold_value)
                except (TypeError, ValueError):
                    continue
                numeric_candidates.append((distance, label))
            if numeric_candidates:
                return min(numeric_candidates)[1]
    elif candidate_type == "copied_operand":
        label = find_family_option("copied_operand")
        if label:
            return label
        for intermediate in main_item.get("expected_intermediates", []):
            try:
                label = resolve_option_for_value(options, intermediate["value"])
            except (KeyError, MCQConsistencyError):
                continue
            if label != gold_option:
                return label
    elif candidate_type == "wrong_operation_order":
        label = find_family_option("wrong_operation_order")
        if label:
            return label

    if normalized_selected:
        try:
            selected_value = resolve_value_for_option(options, normalized_selected)
        except MCQConsistencyError:
            selected_value = ""
        if normalized_selected != gold_option and selected_value:
            return normalized_selected

    for option in options:
        label = str(option["label"]).strip()
        if label != gold_option:
            return label

    raise MCQConsistencyError("Main item does not contain a non-gold option")


def normalize_candidate_against_mcq(
    candidate: Dict[str, Any],
    options: Sequence[Mapping[str, Any]],
    gold_option: str,
    gold_answer: Any,
    candidate_type: str,
    *,
    main_item: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Repair one candidate so its selected option always matches its final answer."""
    validate_mcq_options(options)
    normalized = copy.deepcopy(candidate)
    main_context = dict(main_item or {})
    main_context.setdefault("options", list(options))
    main_context.setdefault("gold_option", str(gold_option).strip())
    main_context.setdefault("gold_final_answer", normalize_mcq_value(gold_answer))

    main_response = normalized.get("main_response")
    if isinstance(main_response, dict):
        repaired_main = copy.deepcopy(main_response)
    else:
        repaired_main = {}

    candidate_text = str(normalized.get("text", ""))
    if "selected_option" not in repaired_main:
        selected_option = extract_main_selected_option(candidate_text)
        if selected_option:
            repaired_main["selected_option"] = selected_option
    if "final_answer" not in repaired_main:
        final_answer = extract_main_final_answer(candidate_text)
        if final_answer:
            repaired_main["final_answer"] = final_answer

    repaired_main["selected_option"] = str(repaired_main.get("selected_option", "")).strip()
    repaired_main["final_answer"] = normalize_mcq_value(repaired_main.get("final_answer", ""))

    if candidate_type in CORRECT_MAIN_FAMILIES:
        repaired_main["selected_option"] = str(gold_option).strip()
        repaired_main["final_answer"] = normalize_mcq_value(gold_answer)
        if candidate_type in FULLY_CORRECT_FAMILIES or candidate_type == "correct_main_wrong_probe":
            repaired_main["reasoning_text"] = main_context.get("gold_solution_text", repaired_main.get("reasoning_text", ""))
            repaired_main["mentioned_intermediates"] = [
                item["value"] for item in main_context.get("expected_intermediates", [])
            ]
            repaired_main["operation_chain"] = list(main_context.get("operation_chain", []))
        else:
            repaired_main.setdefault(
                "reasoning_text",
                "Doğru seçeneği işaretledim ama ara işlem sırasını karıştırdım.",
            )
    elif candidate_type in WRONG_MAIN_FAMILIES:
        mapped_label: str | None = None
        if repaired_main["final_answer"]:
            try:
                mapped_label = resolve_option_for_value(options, repaired_main["final_answer"])
            except MCQConsistencyError:
                mapped_label = None

        if mapped_label and mapped_label != str(gold_option).strip():
            repaired_main["selected_option"] = mapped_label
        else:
            repaired_main["selected_option"] = choose_preferred_wrong_option(
                main_context,
                candidate_type,
                repaired_main.get("selected_option"),
            )
            repaired_main["final_answer"] = resolve_value_for_option(options, repaired_main["selected_option"])

        if candidate_type == "wrong_final":
            repaired_main["reasoning_text"] = f"Sonucu yanlış işlemle {repaired_main['final_answer']} buldum."
        elif candidate_type == "near_miss":
            repaired_main["reasoning_text"] = f"Neredeyse doğru gidip sonucu {repaired_main['final_answer']} yazdım."
        elif candidate_type == "copied_operand":
            repaired_main.setdefault("reasoning_text", "Ara değeri son cevap sanıp doğrudan yazdım.")
        elif candidate_type == "wrong_operation_order":
            repaired_main.setdefault("reasoning_text", "İşlemleri ters sırayla uyguladım.")
    else:
        if repaired_main["final_answer"]:
            repaired_main["selected_option"] = resolve_option_for_value(options, repaired_main["final_answer"])
        elif repaired_main["selected_option"]:
            repaired_main["final_answer"] = resolve_value_for_option(options, repaired_main["selected_option"])
        else:
            raise MCQConsistencyError("Candidate is missing both selected option and final answer")

    normalized["main_response"] = repaired_main
    normalized["text"] = compose_episode_candidate_text(repaired_main, normalized.get("probe_response"))
    validate_candidate_mcq_consistency(
        normalized,
        options,
        gold_option,
        gold_answer,
        candidate_type,
    )
    return normalized


def validate_candidate_mcq_consistency(
    candidate: Mapping[str, Any],
    options: Sequence[Mapping[str, Any]],
    gold_option: str,
    gold_answer: Any,
    candidate_type: str,
) -> None:
    """Assert the candidate satisfies the MCQ invariants required by offline GRPO."""
    validate_mcq_options(options)
    main_response = candidate.get("main_response", candidate)
    if not isinstance(main_response, Mapping):
        raise MCQConsistencyError("Candidate main_response is missing")

    selected_option = str(main_response.get("selected_option", "")).strip()
    final_answer = normalize_mcq_value(main_response.get("final_answer", ""))
    if not selected_option:
        raise MCQConsistencyError("Candidate is missing a selected option")
    if not final_answer:
        raise MCQConsistencyError("Candidate is missing a final answer")

    resolved_option = resolve_option_for_value(options, final_answer)
    if selected_option != resolved_option:
        raise MCQConsistencyError(
            f"Candidate selected option {selected_option!r} does not match final answer {final_answer!r}"
        )

    normalized_gold_option = str(gold_option).strip()
    normalized_gold_answer = normalize_mcq_value(gold_answer)
    if candidate_type in CORRECT_MAIN_FAMILIES:
        if selected_option != normalized_gold_option or final_answer != normalized_gold_answer:
            raise MCQConsistencyError(
                f"Candidate family {candidate_type!r} must keep gold option {normalized_gold_option!r}"
                f" and gold answer {normalized_gold_answer!r}"
            )
    elif candidate_type in WRONG_MAIN_FAMILIES:
        if selected_option == normalized_gold_option or final_answer == normalized_gold_answer:
            raise MCQConsistencyError(
                f"Candidate family {candidate_type!r} must stay on a non-gold main answer"
            )


def normalize_episode_adversarial_candidates(episode: Mapping[str, Any]) -> list[Dict[str, Any]]:
    """Normalize every stored adversarial candidate against one episode's live MCQ."""
    main = episode["main"]
    normalized_candidates: list[Dict[str, Any]] = []
    for candidate in episode.get("adversarial_candidates", []):
        family = str(candidate.get("family", "wrong_final"))
        normalized_candidates.append(
            normalize_candidate_against_mcq(
                candidate,
                main["options"],
                main["gold_option"],
                main["gold_final_answer"],
                family,
                main_item=main,
            )
        )
    return normalized_candidates


def validate_episode_adversarial_candidates(episode: Mapping[str, Any]) -> None:
    """Validate the stored adversarial candidates inside one episode record."""
    main = episode["main"]
    validate_mcq_options(main["options"])
    for candidate in episode.get("adversarial_candidates", []):
        validate_candidate_mcq_consistency(
            candidate,
            main["options"],
            main["gold_option"],
            main["gold_final_answer"],
            str(candidate.get("family", "wrong_final")),
        )
