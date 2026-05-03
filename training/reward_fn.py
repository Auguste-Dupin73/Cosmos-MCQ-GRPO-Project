from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Callable, Mapping, Sequence

from mcq_consistency import normalize_mcq_value
from reasoning_checks import contradictions_detected, final_answer_correct

from training.formatting import extract_prompt_options, parse_episode_completion


@dataclass(frozen=True)
class RewardConfig:
    full_credit: float = 1.0
    main_and_probe_credit: float = 0.7
    main_only_credit: float = 0.4
    correct_option_wrong_reasoning_credit: float = 0.25
    failure_credit: float = 0.0
    format_credit: float = 0.10
    option_present_credit: float = 0.03
    main_final_present_credit: float = 0.03
    probe_final_present_credit: float = 0.03
    reasoning_present_credit: float = 0.03
    numeric_work_credit: float = 0.02
    gold_answer_mentioned_credit: float = 0.04
    max_shaping_credit: float = 0.25
    non_joint_reward_cap: float = 0.95


NUMBER_TOKEN_RE = re.compile(r"-?\d+(?:[.,]\d+)?")


NEGATIVE_REASONING_CUES = (
    "yanlis",
    "yanlış",
    "karistir",
    "karıştır",
    "ters sira",
    "ters sıra",
    "hata yapt",
    "wrong reasoning",
)


def normalize_reward_config(raw_config: Mapping[str, Any] | None = None) -> RewardConfig:
    raw_config = dict(raw_config or {})
    return RewardConfig(
        full_credit=float(raw_config.get("full_credit", 1.0)),
        main_and_probe_credit=float(raw_config.get("main_and_probe_credit", 0.7)),
        main_only_credit=float(raw_config.get("main_only_credit", 0.4)),
        correct_option_wrong_reasoning_credit=float(
            raw_config.get("correct_option_wrong_reasoning_credit", 0.25)
        ),
        failure_credit=float(raw_config.get("failure_credit", 0.0)),
        format_credit=float(raw_config.get("format_credit", 0.10)),
        option_present_credit=float(raw_config.get("option_present_credit", 0.03)),
        main_final_present_credit=float(raw_config.get("main_final_present_credit", 0.03)),
        probe_final_present_credit=float(raw_config.get("probe_final_present_credit", 0.03)),
        reasoning_present_credit=float(raw_config.get("reasoning_present_credit", 0.03)),
        numeric_work_credit=float(raw_config.get("numeric_work_credit", 0.02)),
        gold_answer_mentioned_credit=float(raw_config.get("gold_answer_mentioned_credit", 0.04)),
        max_shaping_credit=float(raw_config.get("max_shaping_credit", 0.25)),
        non_joint_reward_cap=float(raw_config.get("non_joint_reward_cap", 0.95)),
    )


def score_completion_against_episode(
    completion: str,
    *,
    prompt: str | None = None,
    main: Mapping[str, Any] | None = None,
    probe: Mapping[str, Any] | None = None,
    gold: Mapping[str, Any] | None = None,
    reward_spec: Mapping[str, Any] | None = None,
    task_type: str | None = None,
    reward_config: Mapping[str, Any] | RewardConfig | None = None,
) -> dict[str, Any]:
    reward_conf = reward_config if isinstance(reward_config, RewardConfig) else normalize_reward_config(reward_config)
    task_type = (task_type or "episode").strip().lower()
    main_item, probe_item = _coerce_episode_targets(prompt=prompt, main=main, probe=probe, gold=gold)
    parsed = parse_episode_completion(completion)

    selected_option = parsed["main_selected_option"]
    main_reasoning = parsed["main_reasoning_text"]
    main_final = parsed["main_final_answer"]
    probe_reasoning = parsed["probe_reasoning_text"]
    probe_final = parsed["probe_final_answer"]
    gold_option = main_item.get("gold_option")
    gold_main_final = main_item.get("gold_final_answer")
    gold_probe_final = probe_item.get("gold_final_answer")

    option_present = bool(selected_option)
    main_final_present = bool(main_final)
    probe_final_present = bool(probe_final)
    reasoning_present = bool(main_reasoning.strip() or probe_reasoning.strip())
    numeric_work_present = bool(parsed["main_numbers"] or parsed["probe_numbers"] or NUMBER_TOKEN_RE.search(completion))
    gold_answer_mentioned = _gold_answer_mentioned(completion, gold_main_final, gold_probe_final)
    if task_type == "main":
        format_flags = [option_present, bool(main_reasoning.strip()), main_final_present]
    elif task_type == "probe":
        format_flags = [bool(probe_reasoning.strip()), probe_final_present]
    else:
        format_flags = [
            option_present,
            bool(main_reasoning.strip()),
            main_final_present,
            bool(probe_reasoning.strip()),
            probe_final_present,
        ]
    format_compliance = sum(float(flag) for flag in format_flags) / len(format_flags)
    shaping_reward = _compute_shaping_reward(
        format_compliance=format_compliance,
        option_present=option_present,
        main_final_present=main_final_present,
        probe_final_present=probe_final_present,
        reasoning_present=reasoning_present,
        numeric_work_present=numeric_work_present,
        gold_answer_mentioned=gold_answer_mentioned,
        reward_config=reward_conf,
    )

    option_accuracy = bool(selected_option) and bool(gold_option) and selected_option == gold_option
    main_accuracy = bool(main_final) and bool(gold_main_final) and final_answer_correct(main_final, gold_main_final)
    probe_accuracy = bool(probe_final) and bool(gold_probe_final) and final_answer_correct(probe_final, gold_probe_final)

    try:
        contradiction = contradictions_detected(
            {
                "selected_option": selected_option,
                "final_answer": main_final,
                "reasoning_text": main_reasoning,
            },
            main_item,
        )
    except Exception:
        contradiction = False

    expected_intermediates = [normalize_mcq_value(item["value"]) for item in main_item.get("expected_intermediates", [])]
    observed_intermediates = parsed["main_equation_results"] or parsed["main_numbers"]
    intermediates_match = _intermediate_match(observed_intermediates, expected_intermediates, main_reasoning)

    expected_ops = list(main_item.get("operation_chain", []))
    observed_ops = parsed["main_operation_chain"]
    operation_match = _operation_match(observed_ops, expected_ops)

    probe_expected_intermediates = [
        normalize_mcq_value(item["value"]) for item in probe_item.get("expected_intermediates", [])
    ]
    probe_observed_intermediates = parsed["probe_equation_results"] or parsed["probe_numbers"]
    probe_intermediates_match = _intermediate_match(
        probe_observed_intermediates,
        probe_expected_intermediates,
        probe_reasoning,
    )
    probe_expected_ops = list(probe_item.get("operation_chain", []))
    probe_observed_ops = parsed["probe_operation_chain"]
    probe_operation_match = _operation_match(probe_observed_ops, probe_expected_ops)

    reasoning_invalid = contradiction or _has_negative_reasoning_cue(main_reasoning)
    if observed_intermediates and expected_intermediates and not _intermediate_match(
        observed_intermediates,
        expected_intermediates,
        main_reasoning,
    ):
        reasoning_invalid = True
    if observed_ops and expected_ops and not _operation_match(observed_ops, expected_ops):
        reasoning_invalid = True

    main_reasoning_consistent = bool(main_reasoning.strip()) and intermediates_match and operation_match and not contradiction
    probe_reasoning_consistent = bool(probe_reasoning.strip()) and probe_intermediates_match and probe_operation_match
    if task_type == "probe":
        reasoning_consistent = probe_reasoning_consistent
    else:
        reasoning_consistent = main_reasoning_consistent

    correct_option_wrong_reasoning = option_accuracy and main_accuracy and reasoning_invalid
    joint_success = _joint_success(
        option_accuracy=option_accuracy,
        main_accuracy=main_accuracy,
        reasoning_consistent=reasoning_consistent,
        probe_accuracy=probe_accuracy,
        reward_spec=reward_spec,
        task_type=task_type,
    )
    reward = _compute_reward(
        option_accuracy=option_accuracy,
        main_accuracy=main_accuracy,
        reasoning_consistent=reasoning_consistent,
        reasoning_invalid=reasoning_invalid,
        probe_accuracy=probe_accuracy,
        joint_success=joint_success,
        reward_spec=reward_spec,
        reward_config=reward_conf,
        shaping_reward=shaping_reward,
        task_type=task_type,
    )

    return {
        "reward": reward,
        "shaping_reward": shaping_reward,
        "format_compliance": float(format_compliance),
        "option_present": float(option_present),
        "main_final_present": float(main_final_present),
        "probe_final_present": float(probe_final_present),
        "reasoning_present": float(reasoning_present),
        "numeric_work_present": float(numeric_work_present),
        "gold_answer_mentioned": float(gold_answer_mentioned),
        "main_accuracy": float(main_accuracy),
        "option_accuracy": float(option_accuracy),
        "probe_accuracy": float(probe_accuracy),
        "reasoning_consistent": float(reasoning_consistent),
        "joint_success": float(joint_success),
        "correct_option_wrong_reasoning": float(correct_option_wrong_reasoning),
        "parsed": parsed,
        "reasoning_invalid": bool(reasoning_invalid),
        "task_type": task_type,
        "reward_config": asdict(reward_conf),
    }


def build_reward_functions(
    reward_config: Mapping[str, Any] | None = None,
) -> tuple[list[Callable[..., list[float]]], list[float]]:
    reward_conf = normalize_reward_config(reward_config)

    def total_reward(**kwargs) -> list[float]:
        return _batched_metric("reward", reward_conf=reward_conf, **kwargs)

    def main_accuracy_reward(**kwargs) -> list[float]:
        return _batched_metric("main_accuracy", reward_conf=reward_conf, **kwargs)

    def option_accuracy_reward(**kwargs) -> list[float]:
        return _batched_metric("option_accuracy", reward_conf=reward_conf, **kwargs)

    def reasoning_consistency_reward(**kwargs) -> list[float]:
        return _batched_metric("reasoning_consistent", reward_conf=reward_conf, **kwargs)

    def probe_accuracy_reward(**kwargs) -> list[float]:
        return _batched_metric("probe_accuracy", reward_conf=reward_conf, **kwargs)

    def joint_success_reward(**kwargs) -> list[float]:
        return _batched_metric("joint_success", reward_conf=reward_conf, **kwargs)

    def correct_option_wrong_reasoning_reward(**kwargs) -> list[float]:
        return _batched_metric("correct_option_wrong_reasoning", reward_conf=reward_conf, **kwargs)

    def format_compliance_reward(**kwargs) -> list[float]:
        return _batched_metric("format_compliance", reward_conf=reward_conf, **kwargs)

    def shaping_reward(**kwargs) -> list[float]:
        return _batched_metric("shaping_reward", reward_conf=reward_conf, **kwargs)

    def option_present_reward(**kwargs) -> list[float]:
        return _batched_metric("option_present", reward_conf=reward_conf, **kwargs)

    def main_final_present_reward(**kwargs) -> list[float]:
        return _batched_metric("main_final_present", reward_conf=reward_conf, **kwargs)

    def probe_final_present_reward(**kwargs) -> list[float]:
        return _batched_metric("probe_final_present", reward_conf=reward_conf, **kwargs)

    def numeric_work_reward(**kwargs) -> list[float]:
        return _batched_metric("numeric_work_present", reward_conf=reward_conf, **kwargs)

    def gold_answer_mentioned_reward(**kwargs) -> list[float]:
        return _batched_metric("gold_answer_mentioned", reward_conf=reward_conf, **kwargs)

    total_reward.__name__ = "episode_reward"
    main_accuracy_reward.__name__ = "main_accuracy"
    option_accuracy_reward.__name__ = "option_accuracy"
    reasoning_consistency_reward.__name__ = "reasoning_consistency"
    probe_accuracy_reward.__name__ = "probe_accuracy"
    joint_success_reward.__name__ = "joint_success"
    correct_option_wrong_reasoning_reward.__name__ = "correct_option_wrong_reasoning"
    format_compliance_reward.__name__ = "format_compliance"
    shaping_reward.__name__ = "shaping_reward"
    option_present_reward.__name__ = "option_present"
    main_final_present_reward.__name__ = "main_final_present"
    probe_final_present_reward.__name__ = "probe_final_present"
    numeric_work_reward.__name__ = "numeric_work_present"
    gold_answer_mentioned_reward.__name__ = "gold_answer_mentioned"

    reward_funcs = [
        total_reward,
        main_accuracy_reward,
        option_accuracy_reward,
        reasoning_consistency_reward,
        probe_accuracy_reward,
        joint_success_reward,
        correct_option_wrong_reasoning_reward,
        format_compliance_reward,
        shaping_reward,
        option_present_reward,
        main_final_present_reward,
        probe_final_present_reward,
        numeric_work_reward,
        gold_answer_mentioned_reward,
    ]
    reward_weights = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    return reward_funcs, reward_weights


def _batched_metric(
    metric_name: str,
    *,
    prompts: Sequence[str],
    completions: Sequence[str],
    main: Sequence[Mapping[str, Any]] | None = None,
    probe: Sequence[Mapping[str, Any]] | None = None,
    gold: Sequence[Mapping[str, Any]] | None = None,
    reward_spec: Sequence[Mapping[str, Any]] | None = None,
    task_type: Sequence[str] | None = None,
    reward_conf: RewardConfig,
    **_: Any,
) -> list[float]:
    results: list[float] = []
    main = list(main or [None] * len(completions))
    probe = list(probe or [None] * len(completions))
    gold = list(gold or [None] * len(completions))
    reward_spec = list(reward_spec or [{}] * len(completions))
    task_type = list(task_type or ["episode"] * len(completions))

    for prompt, completion, main_item, probe_item, gold_item, reward_item, task_item in zip(
        prompts, completions, main, probe, gold, reward_spec, task_type, strict=True
    ):
        score = score_completion_against_episode(
            completion,
            prompt=prompt,
            main=main_item,
            probe=probe_item,
            gold=gold_item,
            reward_spec=reward_item,
            task_type=task_item,
            reward_config=reward_conf,
        )
        results.append(float(score[metric_name]))
    return results


def _joint_success(
    *,
    option_accuracy: bool,
    main_accuracy: bool,
    reasoning_consistent: bool,
    probe_accuracy: bool,
    reward_spec: Mapping[str, Any] | None,
    task_type: str,
) -> bool:
    if task_type == "main":
        return bool(main_accuracy and option_accuracy and reasoning_consistent)
    if task_type == "probe":
        return bool(probe_accuracy and reasoning_consistent)

    reward_spec = dict(reward_spec or {})
    option_ok = option_accuracy if reward_spec.get("require_main_option_correct", True) else True
    reasoning_ok = reasoning_consistent if reward_spec.get("require_main_reasoning_consistent", True) else True
    probe_ok = probe_accuracy if reward_spec.get("require_probe_correct", True) else True
    return bool(main_accuracy and option_ok and reasoning_ok and probe_ok)


def _compute_reward(
    *,
    option_accuracy: bool,
    main_accuracy: bool,
    reasoning_consistent: bool,
    reasoning_invalid: bool,
    probe_accuracy: bool,
    joint_success: bool,
    reward_spec: Mapping[str, Any] | None,
    reward_config: RewardConfig,
    shaping_reward: float,
    task_type: str,
) -> float:
    if joint_success:
        return reward_config.full_credit
    if task_type == "probe":
        if probe_accuracy:
            return _cap_non_joint_reward(reward_config.main_only_credit + shaping_reward, reward_config)
        return _cap_non_joint_reward(reward_config.failure_credit + shaping_reward, reward_config)

    if option_accuracy and main_accuracy and reasoning_invalid:
        return _cap_non_joint_reward(reward_config.correct_option_wrong_reasoning_credit + shaping_reward, reward_config)

    reward_spec = dict(reward_spec or {})
    option_ok = option_accuracy if reward_spec.get("require_main_option_correct", True) else True
    probe_ok = probe_accuracy if reward_spec.get("require_probe_correct", True) else True

    if main_accuracy and option_ok and probe_ok:
        return _cap_non_joint_reward(reward_config.main_and_probe_credit + shaping_reward, reward_config)
    if main_accuracy and option_ok:
        return _cap_non_joint_reward(reward_config.main_only_credit + shaping_reward, reward_config)
    return _cap_non_joint_reward(reward_config.failure_credit + shaping_reward, reward_config)


def _compute_shaping_reward(
    *,
    format_compliance: float,
    option_present: bool,
    main_final_present: bool,
    probe_final_present: bool,
    reasoning_present: bool,
    numeric_work_present: bool,
    gold_answer_mentioned: bool,
    reward_config: RewardConfig,
) -> float:
    reward = reward_config.format_credit * format_compliance
    if option_present:
        reward += reward_config.option_present_credit
    if main_final_present:
        reward += reward_config.main_final_present_credit
    if probe_final_present:
        reward += reward_config.probe_final_present_credit
    if reasoning_present:
        reward += reward_config.reasoning_present_credit
    if numeric_work_present:
        reward += reward_config.numeric_work_credit
    if gold_answer_mentioned:
        reward += reward_config.gold_answer_mentioned_credit
    return min(reward, reward_config.max_shaping_credit)


def _cap_non_joint_reward(reward: float, reward_config: RewardConfig) -> float:
    return min(reward, reward_config.non_joint_reward_cap)


def _coerce_episode_targets(
    *,
    prompt: str | None,
    main: Mapping[str, Any] | None,
    probe: Mapping[str, Any] | None,
    gold: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    main_item = dict(main or {})
    probe_item = dict(probe or {})
    gold = dict(gold or {})

    if "gold_option" not in main_item and "main_gold_option" in gold:
        main_item["gold_option"] = gold["main_gold_option"]
    if "gold_final_answer" not in main_item and "main_gold_final_answer" in gold:
        main_item["gold_final_answer"] = gold["main_gold_final_answer"]
    if "gold_final_answer" not in probe_item and "probe_gold_final_answer" in gold:
        probe_item["gold_final_answer"] = gold["probe_gold_final_answer"]
    if "options" not in main_item:
        main_item["options"] = extract_prompt_options(prompt or "")
    if "expected_intermediates" not in main_item:
        main_item["expected_intermediates"] = []
    if "operation_chain" not in main_item:
        main_item["operation_chain"] = []
    if "expected_intermediates" not in probe_item:
        probe_item["expected_intermediates"] = []
    if "operation_chain" not in probe_item:
        probe_item["operation_chain"] = []
    return main_item, probe_item


def _intermediate_match(
    observed: Sequence[str],
    expected: Sequence[str],
    reasoning_text: str,
) -> bool:
    if expected:
        return _contains_subsequence(observed, expected) or bool(set(observed).intersection(expected))
    return bool(reasoning_text.strip())


def _operation_match(observed: Sequence[str], expected: Sequence[str]) -> bool:
    if expected and observed:
        return _contains_subsequence(expected, observed)
    if expected and not observed:
        return False
    return True


def _contains_subsequence(observed: Sequence[str], expected: Sequence[str]) -> bool:
    if not expected:
        return True
    expected_index = 0
    for item in observed:
        if item == expected[expected_index]:
            expected_index += 1
            if expected_index == len(expected):
                return True
    return False


def _has_negative_reasoning_cue(text: str) -> bool:
    lowered = text.casefold()
    return any(cue in lowered for cue in NEGATIVE_REASONING_CUES)


def _gold_answer_mentioned(text: str, *gold_values: Any) -> bool:
    number_tokens = {
        normalize_mcq_value(match.group(0).replace(",", "."))
        for match in NUMBER_TOKEN_RE.finditer(text or "")
    }
    lowered = (text or "").casefold()
    for raw_gold in gold_values:
        if raw_gold is None:
            continue
        gold = normalize_mcq_value(str(raw_gold).replace(",", "."))
        if gold and gold in number_tokens:
            return True
        if gold and not NUMBER_TOKEN_RE.fullmatch(gold) and gold.casefold() in lowered:
            return True
    return False
