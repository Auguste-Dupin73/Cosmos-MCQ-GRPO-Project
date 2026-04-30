from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

from mcq_consistency import compose_episode_candidate_text, normalize_mcq_value


OPTION_LINE_RE = re.compile(r"(?m)^\s*([A-Z])\)\s*(.+?)\s*$")
MARKDOWN_LABEL_PREFIX = r"(?:#{1,6}\s*)?(?:\*\*)?\s*"
MARKDOWN_LABEL_SUFFIX = r"\s*(?:\*\*)?\s*:"
MAIN_OPTION_LABEL = r"Ana\s+soru\s+se(?:\u00e7|c)imi"
MAIN_REASONING_LABEL = r"Ana\s+(?:\u00e7\u00f6z\u00fcm|cozum)"
MAIN_FINAL_LABEL = r"Ana\s+nihai\s+cevap"
PROBE_REASONING_LABEL = r"Probe\s+(?:\u00e7\u00f6z\u00fcm|cozum)"
PROBE_FINAL_LABEL = r"Probe\s+nihai\s+cevap"
ANY_RESPONSE_LABEL_RE = re.compile(
    rf"(?im)^\s*{MARKDOWN_LABEL_PREFIX}"
    rf"(?:{MAIN_OPTION_LABEL}|{MAIN_REASONING_LABEL}|{MAIN_FINAL_LABEL}|"
    rf"{PROBE_REASONING_LABEL}|{PROBE_FINAL_LABEL}|Probe\s+soru(?:\s+se(?:\u00e7|c)imi)?)"
    rf"(?:{MARKDOWN_LABEL_SUFFIX}|\s*$)"
)
OPTION_PLACEHOLDER_RE = re.compile(r"<?\s*A\s*/\s*B\s*/\s*C\s*/\s*D\s*>?", re.IGNORECASE)
OPTION_VALUE_RE = re.compile(r"(?im)(?:^|[\s*_`])([A-D])\s*[\).]")
STANDALONE_OPTION_RE = re.compile(r"(?i)^[*_`\s]*([A-D])[*_`\s]*$")
NUMBER_RE = re.compile(r"-?\d+(?:[.,]\d+)?")
EQUATION_RESULT_RE = re.compile(r"=\s*(-?\d+(?:[.,]\d+)?)")
OPERATOR_RE = re.compile(r"[+\-*/xX]")


RESPONSE_FORMAT_BLOCK = "\n".join(
    [
        "",
        "Cikti tam 5 satir olmali. Markdown, ###, kalin yazi, secenek listesi veya yeni soru metni yazma.",
        "Her ':' sonrasini kendi cevabinla doldur; <A/B/C/D> gibi sablon metni kopyalama.",
        "Ana soru secimi: sadece A/B/C/D harfi",
        "Ana cozum: tek kisa islem veya gerekce",
        "Ana nihai cevap: sadece sayisal deger",
        "Probe cozum: tek kisa islem veya gerekce; Probe soru secimi yazma",
        "Probe nihai cevap: sadece sayisal deger",
    ]
)


def render_option_lines(options: Sequence[Mapping[str, Any]]) -> list[str]:
    return [f"{option['label']}) {option['text']}" for option in options]


def build_episode_prompt(
    episode: Mapping[str, Any],
    *,
    include_support_pack: bool = False,
    append_response_format: bool = False,
) -> str:
    main = episode["main"]
    probe = episode["probe"]
    lines = [
        "Ana \u00e7oktan se\u00e7meli matematik sorusunu \u00e7\u00f6z.",
        main["mcq_stem"],
        *render_option_lines(main["options"]),
        "Do\u011fru se\u00e7ene\u011fi belirt, \u00e7\u00f6z\u00fcm\u00fcn\u00fc a\u00e7\u0131kla ve nihai cevab\u0131 yaz.",
        "Ana soru do\u011fruysa ayn\u0131 beceriyi \u00f6l\u00e7en probe sorusunu da \u00e7\u00f6z.",
        f"Probe soru: {probe['question_text']}",
    ]
    if include_support_pack:
        pack = episode["support_pack"]
        lines.extend(
            [
                "\u0130pucu paketi:",
                f"- \u00d6zet: {pack['skill_summary']}",
                *[f"- Kural: {hint}" for hint in pack["formula_hints"]],
                f"- Mini \u00f6rnek: {pack['mini_example']}",
            ]
        )
    prompt = "\n".join(lines)
    if append_response_format:
        prompt = f"{prompt}\n{RESPONSE_FORMAT_BLOCK}"
    return prompt


def build_prompt_from_record(
    row: Mapping[str, Any],
    *,
    include_support_pack: bool | None = None,
    append_response_format: bool = False,
) -> str:
    if "main" in row and "probe" in row and "support_pack" in row:
        return build_episode_prompt(
            row,
            include_support_pack=bool(include_support_pack),
            append_response_format=append_response_format,
        )
    prompt = str(row["prompt"])
    if append_response_format and RESPONSE_FORMAT_BLOCK not in prompt:
        return f"{prompt}\n{RESPONSE_FORMAT_BLOCK}"
    return prompt


def extract_prompt_options(prompt: str) -> list[dict[str, str]]:
    return [{"label": match.group(1), "text": match.group(2).strip()} for match in OPTION_LINE_RE.finditer(prompt)]


def _clean_scalar(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = _strip_inline_markup(value.strip()).rstrip(".")
    return cleaned or None


def _strip_inline_markup(value: str) -> str:
    cleaned = value.strip()
    cleaned = re.sub(r"(?m)^\s*#{1,6}\s*", "", cleaned)
    cleaned = re.sub(r"^\*\*(.*?)\*\*$", r"\1", cleaned, flags=re.DOTALL)
    return cleaned.strip().strip("*_`").strip()


def _first_nonempty_line(value: str | None) -> str | None:
    if value is None:
        return None
    for line in value.splitlines():
        cleaned = _clean_scalar(line)
        if cleaned:
            return cleaned
    return None


def _extract_labeled_section(text: str, label_pattern: str) -> str | None:
    label_re = re.compile(rf"(?im)^\s*{MARKDOWN_LABEL_PREFIX}{label_pattern}{MARKDOWN_LABEL_SUFFIX}\s*")
    match = label_re.search(text)
    if not match:
        return None
    next_match = ANY_RESPONSE_LABEL_RE.search(text, match.end())
    end = next_match.start() if next_match else len(text)
    return _clean_scalar(text[match.end() : end])


def _extract_option_value(value: str | None) -> str | None:
    cleaned = _clean_scalar(value)
    if not cleaned:
        return None
    cleaned = OPTION_PLACEHOLDER_RE.sub("", cleaned).strip()
    if not cleaned:
        return None

    standalone = STANDALONE_OPTION_RE.fullmatch(cleaned)
    if standalone:
        return standalone.group(1).upper()

    labels = [match.group(1).upper() for match in OPTION_VALUE_RE.finditer(cleaned)]
    if len(set(labels)) == 1:
        return labels[0]
    return None


def _clean_final_answer(value: str | None) -> str | None:
    cleaned = _first_nonempty_line(value)
    if not cleaned:
        return None
    cleaned = re.sub(r"(?i)^[A-D]\s*[\).:-]\s*", "", cleaned).strip()
    cleaned = _strip_inline_markup(cleaned).rstrip(".")
    return cleaned or None


def _extract_numbers(text: str) -> list[str]:
    return [normalize_mcq_value(match.group(0).replace(",", ".")) for match in NUMBER_RE.finditer(text or "")]


def _extract_equation_results(text: str) -> list[str]:
    return [normalize_mcq_value(match.group(1).replace(",", ".")) for match in EQUATION_RESULT_RE.finditer(text or "")]


def _infer_operation_chain(text: str) -> list[str]:
    mapping = {
        "+": "add",
        "-": "subtract",
        "*": "multiply",
        "x": "multiply",
        "X": "multiply",
        "/": "divide",
    }
    return [mapping[token.group(0)] for token in OPERATOR_RE.finditer(text or "")]


def parse_episode_completion(text: str) -> dict[str, Any]:
    raw_text = text.strip()
    main_option = _extract_option_value(_extract_labeled_section(raw_text, MAIN_OPTION_LABEL))
    main_reasoning = _extract_labeled_section(raw_text, MAIN_REASONING_LABEL) or ""
    main_final = _clean_final_answer(_extract_labeled_section(raw_text, MAIN_FINAL_LABEL))
    probe_reasoning = _extract_labeled_section(raw_text, PROBE_REASONING_LABEL) or ""
    probe_final = _clean_final_answer(_extract_labeled_section(raw_text, PROBE_FINAL_LABEL))
    main_results = _extract_equation_results(main_reasoning)
    probe_results = _extract_equation_results(probe_reasoning)

    return {
        "raw_text": raw_text,
        "main_selected_option": main_option,
        "main_reasoning_text": main_reasoning,
        "main_final_answer": main_final,
        "main_numbers": _extract_numbers(main_reasoning),
        "main_equation_results": main_results,
        "main_operation_chain": _infer_operation_chain(main_reasoning),
        "probe_reasoning_text": probe_reasoning,
        "probe_final_answer": probe_final,
        "probe_numbers": _extract_numbers(probe_reasoning),
        "probe_equation_results": probe_results,
        "probe_operation_chain": _infer_operation_chain(probe_reasoning),
    }


def derive_episode_targets_from_text(prompt: str, response_text: str) -> dict[str, Any]:
    parsed = parse_episode_completion(response_text)
    options = extract_prompt_options(prompt)
    main_results = parsed["main_equation_results"]
    probe_results = parsed["probe_equation_results"]

    main = {
        "options": options,
        "gold_option": parsed["main_selected_option"],
        "gold_final_answer": parsed["main_final_answer"],
        "expected_intermediates": [
            {"name": f"main_step_{index + 1}", "value": value}
            for index, value in enumerate(main_results)
        ],
        "operation_chain": parsed["main_operation_chain"],
    }
    probe = {
        "gold_final_answer": parsed["probe_final_answer"],
        "expected_intermediates": [
            {"name": f"probe_step_{index + 1}", "value": value}
            for index, value in enumerate(probe_results)
        ],
        "operation_chain": parsed["probe_operation_chain"],
    }
    return {"main": main, "probe": probe}


def canonical_candidate_text(
    *,
    main_option: str,
    main_reasoning: str,
    main_final_answer: str,
    probe_reasoning: str | None = None,
    probe_final_answer: str | None = None,
) -> str:
    probe_response = None
    if probe_final_answer is not None:
        probe_response = {
            "reasoning_text": probe_reasoning or "",
            "final_answer": probe_final_answer,
        }
    return compose_episode_candidate_text(
        {
            "selected_option": main_option,
            "reasoning_text": main_reasoning,
            "final_answer": main_final_answer,
        },
        probe_response,
    )


def _match_text(pattern: re.Pattern[str], text: str) -> str | None:
    match = pattern.search(text)
    if not match:
        return None
    return match.group(1).strip()
