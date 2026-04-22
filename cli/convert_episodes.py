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


def convert_episode_rows(episodes: Iterable[Dict[str, Any]], out_dir: Path) -> None:
    """Write all converted episode outputs."""
    out_dir.mkdir(parents=True, exist_ok=True)

    sft_rows = []
    dpo_rows = []
    grpo_offline_rows = []
    grpo_online_rows = []

    for episode in episodes:
        prompt = build_episode_prompt(episode)
        prompt_with_support = build_episode_prompt(episode, include_support_pack=True)
        normalized_candidates = _normalized_episode_candidates(episode)
        gold = normalized_candidates[0]
        rejected = normalized_candidates[1]

        sft_rows.append(
            {
                "id": episode["episode_id"],
                "prompt": prompt,
                "answer": gold["text"],
                "tags": episode["tags"],
            }
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
        grpo_offline_rows.append(
            {
                "id": episode["episode_id"],
                "prompt": prompt,
                "responses": _offline_candidates(episode, normalized_candidates),
                "reward_spec": episode["reward_spec"],
                "tags": episode["tags"],
            }
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
