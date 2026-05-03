"""Convert episode JSONL into SFT, DPO, and GRPO episode datasets."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Iterable

try:
    from cli._bootstrap import ensure_repo_root
except ImportError:  # pragma: no cover
    from _bootstrap import ensure_repo_root

ensure_repo_root()

from adversarial_negatives import build_gold_episode_response
from mcq_consistency import normalize_episode_adversarial_candidates
from reasoning_checks import evaluate_main_response
from training.formatting import build_main_prompt, build_probe_prompt
from utils_episode import load_jsonl, resolve_repo_path, write_jsonl


def build_episode_prompt(episode: Dict[str, Any], include_support_pack: bool = False) -> str:
    """Build a prompt for episode-style training examples."""
    main = episode["main"]
    probe = episode["probe"]
    option_lines = [f"{option['label']}) {option['text']}" for option in main["options"]]
    lines = [
        "Ana çoktan seçmeli matematik sorusunu çöz.",
        main["mcq_stem"],
        *option_lines,
        "Doğru seçeneği belirt, çözümünü açıkla ve nihai cevabı yaz.",
        "Ana soru doğruysa aynı beceriyi ölçen probe sorusunu da çöz.",
        f"Probe soru: {probe['question_text']}",
    ]
    if include_support_pack:
        pack = episode["support_pack"]
        lines.extend(
            [
                "İpucu paketi:",
                f"- Özet: {pack['skill_summary']}",
                *[f"- Kural: {hint}" for hint in pack["formula_hints"]],
                f"- Mini örnek: {pack['mini_example']}",
            ]
        )
    return "\n".join(lines)


def _score_candidate(candidate: Dict[str, Any], episode: Dict[str, Any]) -> float:
    checks = evaluate_main_response(candidate["main_response"], episode["main"])
    probe = candidate.get("probe_response")
    probe_correct = bool(probe) and str(probe.get("final_answer")) == episode["probe"]["gold_final_answer"]

    if (
        checks["selected_option_correct"]
        and checks["final_answer_correct"]
        and checks["mentioned_intermediates_consistent"]
        and checks["operation_order_consistent"]
        and not checks["contradictions_detected"]
        and probe_correct
    ):
        return 1.0
    if checks["correct_option_wrong_reasoning"]:
        return 0.25
    if checks["selected_option_correct"] and checks["final_answer_correct"] and probe_correct:
        return 0.7
    if checks["selected_option_correct"] and checks["final_answer_correct"]:
        return 0.4
    return 0.0


def _normalized_episode_candidates(episode: Dict[str, Any]) -> list[Dict[str, Any]]:
    return [build_gold_episode_response(episode), *normalize_episode_adversarial_candidates(episode)]


def _offline_candidates(episode: Dict[str, Any], candidates: list[Dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"text": candidate["text"], "score": _score_candidate(candidate, episode)} for candidate in candidates]


def _main_response_text(response: Dict[str, Any]) -> str:
    return "\n".join(
        [
            "<reasoning>",
            str(response.get("reasoning_text", "")),
            "</reasoning>",
            "<final>",
            f"option: {response.get('selected_option', '')}",
            f"main: {response.get('final_answer', '')}",
            "</final>",
        ]
    )


def _probe_response_text(response: Dict[str, Any]) -> str:
    return "\n".join(
        [
            "<reasoning>",
            str(response.get("reasoning_text", "")),
            "</reasoning>",
            "<final>",
            f"probe: {response.get('final_answer', '')}",
            "</final>",
        ]
    )


def _score_main_candidate(candidate: Dict[str, Any], episode: Dict[str, Any]) -> float:
    checks = evaluate_main_response(candidate["main_response"], episode["main"])
    if (
        checks["selected_option_correct"]
        and checks["final_answer_correct"]
        and checks["mentioned_intermediates_consistent"]
        and checks["operation_order_consistent"]
        and not checks["contradictions_detected"]
    ):
        return 1.0
    if checks["correct_option_wrong_reasoning"]:
        return 0.25
    if checks["selected_option_correct"] and checks["final_answer_correct"]:
        return 0.4
    return 0.0


def _score_probe_candidate(candidate: Dict[str, Any], episode: Dict[str, Any]) -> float:
    response = candidate.get("probe_response")
    if not response:
        return 0.0
    return 1.0 if str(response.get("final_answer")) == str(episode["probe"]["gold_final_answer"]) else 0.0


def _split_offline_candidates(
    episode: Dict[str, Any],
    candidates: list[Dict[str, Any]],
    *,
    task_type: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        if task_type == "main":
            row = {"text": _main_response_text(candidate["main_response"]), "score": _score_main_candidate(candidate, episode)}
        elif candidate.get("probe_response"):
            row = {"text": _probe_response_text(candidate["probe_response"]), "score": _score_probe_candidate(candidate, episode)}
        else:
            continue
        previous = seen.get(row["text"])
        if previous is None or float(row["score"]) > float(previous["score"]):
            seen[row["text"]] = row
    rows.extend(seen.values())
    return rows


def convert_episode_rows(episodes: Iterable[Dict[str, Any]], out_dir: Path) -> None:
    """Write all converted episode outputs."""
    out_dir.mkdir(parents=True, exist_ok=True)

    sft_rows = []
    dpo_rows = []
    grpo_offline_rows = []
    grpo_online_rows = []
    split_sft_rows = []
    split_grpo_offline_rows = []
    split_grpo_online_rows = []

    for episode in episodes:
        prompt = build_episode_prompt(episode)
        prompt_with_support = build_episode_prompt(episode, include_support_pack=True)
        main_prompt = build_main_prompt(episode, append_response_format=True)
        probe_prompt = build_probe_prompt(episode, append_response_format=True)
        normalized_candidates = _normalized_episode_candidates(episode)
        gold = normalized_candidates[0]
        rejected = normalized_candidates[1]
        main_gold_text = _main_response_text(gold["main_response"])
        probe_gold_text = _probe_response_text(gold["probe_response"])

        sft_rows.append(
            {
                "id": episode["episode_id"],
                "prompt": prompt,
                "answer": gold["text"],
                "tags": episode["tags"],
            }
        )
        split_sft_rows.extend(
            [
                {
                    "id": f"{episode['episode_id']}::main",
                    "episode_id": episode["episode_id"],
                    "task_type": "main",
                    "prompt": main_prompt,
                    "answer": main_gold_text,
                    "tags": [*episode["tags"], "task:main"],
                },
                {
                    "id": f"{episode['episode_id']}::probe",
                    "episode_id": episode["episode_id"],
                    "task_type": "probe",
                    "prompt": probe_prompt,
                    "answer": probe_gold_text,
                    "tags": [*episode["tags"], "task:probe"],
                },
            ]
        )
        dpo_rows.append(
            {
                "id": episode["episode_id"],
                "prompt": prompt,
                "chosen": gold["text"],
                "rejected": rejected["text"],
                "tags": episode["tags"],
            }
        )
        split_grpo_offline_rows.extend(
            [
                {
                    "id": f"{episode['episode_id']}::main",
                    "episode_id": episode["episode_id"],
                    "task_type": "main",
                    "prompt": main_prompt,
                    "responses": _split_offline_candidates(episode, normalized_candidates, task_type="main"),
                    "reward_spec": episode["reward_spec"],
                    "tags": [*episode["tags"], "task:main"],
                },
                {
                    "id": f"{episode['episode_id']}::probe",
                    "episode_id": episode["episode_id"],
                    "task_type": "probe",
                    "prompt": probe_prompt,
                    "responses": _split_offline_candidates(episode, normalized_candidates, task_type="probe"),
                    "reward_spec": episode["reward_spec"],
                    "tags": [*episode["tags"], "task:probe"],
                },
            ]
        )
        grpo_offline_rows.append(
            {
                "id": episode["episode_id"],
                "prompt": prompt,
                "responses": _offline_candidates(episode, normalized_candidates),
                "reward_spec": episode["reward_spec"],
                "tags": episode["tags"],
            }
        )
        split_grpo_online_rows.extend(
            [
                {
                    "id": f"{episode['episode_id']}::main",
                    "episode_id": episode["episode_id"],
                    "task_type": "main",
                    "prompt": main_prompt,
                    "gold": {
                        "main_gold_option": episode["main"]["gold_option"],
                        "main_gold_final_answer": episode["main"]["gold_final_answer"],
                        "probe_gold_final_answer": episode["probe"]["gold_final_answer"],
                    },
                    "reward_spec": episode["reward_spec"],
                    "tags": [*episode["tags"], "task:main"],
                },
                {
                    "id": f"{episode['episode_id']}::probe",
                    "episode_id": episode["episode_id"],
                    "task_type": "probe",
                    "prompt": probe_prompt,
                    "gold": {
                        "main_gold_option": episode["main"]["gold_option"],
                        "main_gold_final_answer": episode["main"]["gold_final_answer"],
                        "probe_gold_final_answer": episode["probe"]["gold_final_answer"],
                    },
                    "reward_spec": episode["reward_spec"],
                    "tags": [*episode["tags"], "task:probe"],
                },
            ]
        )
        grpo_online_rows.append(
            {
                "id": episode["episode_id"],
                "prompt": prompt_with_support,
                "gold": {
                    "main_gold_option": episode["main"]["gold_option"],
                    "main_gold_final_answer": episode["main"]["gold_final_answer"],
                    "probe_gold_final_answer": episode["probe"]["gold_final_answer"],
                },
                "reward_spec": episode["reward_spec"],
                "tags": episode["tags"],
            }
        )

    write_jsonl(out_dir / "episode_sft.jsonl", sft_rows)
    write_jsonl(out_dir / "episode_dpo.jsonl", dpo_rows)
    write_jsonl(out_dir / "episode_grpo_offline.jsonl", grpo_offline_rows)
    write_jsonl(out_dir / "episode_grpo_online.jsonl", grpo_online_rows)
    write_jsonl(out_dir / "episode_sft_split.jsonl", split_sft_rows)
    write_jsonl(out_dir / "episode_grpo_offline_split.jsonl", split_grpo_offline_rows)
    write_jsonl(out_dir / "episode_grpo_online_split.jsonl", split_grpo_online_rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    input_path = resolve_repo_path(args.input)
    episodes = load_jsonl(input_path)
    out_dir = Path(args.out_dir)
    convert_episode_rows(episodes, out_dir)
    print(f"Wrote converted episode datasets to {out_dir}")


if __name__ == "__main__":
    main()
