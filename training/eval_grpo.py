from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.dataset import load_episode_records
from training.reward_fn import score_completion_against_episode
from training.utils import (
    coerce_path_list,
    coerce_torch_dtype,
    load_yaml_config,
    resolve_output_path,
    summarize_metric_groups,
    summarize_metric_rows,
    take_limit,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a model or checkpoint on episode-style GRPO data.")
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint or model path to evaluate.")
    parser.add_argument("--data", action="append", default=None, help="Optional override for eval data path(s).")
    parser.add_argument("--output", default=None, help="Optional JSON report output path.")
    parser.add_argument("--predictions_out", default=None, help="Optional JSONL predictions output path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)

    model_cfg = dict(config.get("model", {}))
    data_cfg = dict(config.get("data", {}))
    eval_cfg = dict(config.get("eval", {}))
    reward_cfg = dict(config.get("reward", {}))

    model_name_or_path = args.checkpoint or model_cfg["name_or_path"]
    records = load_eval_records(data_cfg, override_paths=args.data)
    if not records:
        raise ValueError("No evaluation records were loaded")

    tokenizer = load_tokenizer(model_cfg, model_name_or_path)
    model = load_model(model_cfg, model_name_or_path)
    device = torch.device("cuda" if torch.cuda.is_available() and not eval_cfg.get("force_cpu", False) else "cpu")
    model.to(device)
    model.eval()

    batch_size = int(eval_cfg.get("batch_size", 1))
    max_new_tokens = int(eval_cfg.get("max_new_tokens", 192))
    generation_kwargs = build_generation_kwargs(eval_cfg, tokenizer)

    rows = evaluate_records(
        model=model,
        tokenizer=tokenizer,
        records=records,
        device=device,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        reward_cfg=reward_cfg,
        generation_kwargs=generation_kwargs,
    )
    report = {
        "model_name_or_path": str(model_name_or_path),
        "num_examples": len(rows),
        "overall": summarize_metric_rows(rows),
        "by_skill": summarize_metric_groups(rows, "skill_id"),
        "by_tier": summarize_metric_groups(rows, "tier"),
        "by_template": summarize_metric_groups(rows, "template_id"),
    }

    print(json.dumps(report, ensure_ascii=False, indent=2))

    if args.output:
        output_path = resolve_output_path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.predictions_out:
        write_jsonl(resolve_output_path(args.predictions_out), rows)


def load_eval_records(data_cfg: dict[str, Any], *, override_paths: list[str] | None) -> list[dict[str, Any]]:
    dataset_format = data_cfg.get("dataset_format", "auto")
    include_support_pack = bool(data_cfg.get("include_support_pack", True))
    append_response_format = bool(data_cfg.get("append_response_format", False))

    if override_paths:
        paths = override_paths
    else:
        paths = [str(path) for path in coerce_path_list(data_cfg.get("eval_paths") or data_cfg.get("eval_path"))]
        if not paths:
            paths = [str(path) for path in coerce_path_list(data_cfg.get("train_paths") or data_cfg.get("train_path"))]

    records = load_episode_records(
        paths,
        dataset_format=dataset_format,
        include_support_pack=include_support_pack,
        append_response_format=append_response_format,
        max_samples=None,
        shuffle=False,
        seed=int(data_cfg.get("seed", 42) or 42),
    )
    return take_limit(records, data_cfg.get("max_eval_samples"))


def load_tokenizer(model_cfg: dict[str, Any], model_name_or_path: str):
    tokenizer_name_or_path = model_cfg.get("tokenizer_name_or_path") or model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name_or_path,
        trust_remote_code=bool(model_cfg.get("trust_remote_code", False)),
        local_files_only=bool(model_cfg.get("local_files_only", False)),
    )
    tokenizer.truncation_side = "left"
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    if tokenizer.pad_token is None:  # pragma: no cover
        raise ValueError("Tokenizer is missing pad_token, eos_token, and unk_token")
    return tokenizer


def load_model(model_cfg: dict[str, Any], model_name_or_path: str):
    model_kwargs = dict(model_cfg.get("model_init_kwargs", {}))
    if "torch_dtype" in model_cfg:
        model_kwargs["torch_dtype"] = coerce_torch_dtype(model_cfg["torch_dtype"])
    if "trust_remote_code" in model_cfg:
        model_kwargs["trust_remote_code"] = bool(model_cfg["trust_remote_code"])
    if "local_files_only" in model_cfg:
        model_kwargs["local_files_only"] = bool(model_cfg["local_files_only"])
    return AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)


def build_generation_kwargs(eval_cfg: dict[str, Any], tokenizer) -> dict[str, Any]:
    do_sample = bool(eval_cfg.get("do_sample", False))
    kwargs: dict[str, Any] = {
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        kwargs["temperature"] = float(eval_cfg.get("temperature", 0.8))
        kwargs["top_p"] = float(eval_cfg.get("top_p", 1.0))
        kwargs["top_k"] = int(eval_cfg.get("top_k", 0))
    return kwargs


def evaluate_records(
    *,
    model,
    tokenizer,
    records: list[dict[str, Any]],
    device: torch.device,
    batch_size: int,
    max_new_tokens: int,
    reward_cfg: dict[str, Any],
    generation_kwargs: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for start in range(0, len(records), batch_size):
            batch = records[start : start + batch_size]
            prompts = [record["prompt"] for record in batch]
            encoded = tokenizer(prompts, padding=True, return_tensors="pt")
            encoded = {key: value.to(device) for key, value in encoded.items()}
            generated = model.generate(**encoded, max_new_tokens=max_new_tokens, **generation_kwargs)
            prompt_length = encoded["input_ids"].shape[1]
            completions = tokenizer.batch_decode(generated[:, prompt_length:], skip_special_tokens=True)

            for record, completion in zip(batch, completions, strict=True):
                score = score_completion_against_episode(
                    completion,
                    prompt=record["prompt"],
                    main=record.get("main"),
                    probe=record.get("probe"),
                    gold=record.get("gold"),
                    reward_spec=record.get("reward_spec"),
                    reward_config=reward_cfg,
                )
                rows.append(
                    {
                        "id": record["id"],
                        "skill_id": record.get("skill_id"),
                        "template_id": record.get("template_id"),
                        "tier": record.get("tier"),
                        "reward": score["reward"],
                        "main_accuracy": score["main_accuracy"],
                        "option_accuracy": score["option_accuracy"],
                        "probe_accuracy": score["probe_accuracy"],
                        "joint_success": score["joint_success"],
                        "reasoning_consistent": score["reasoning_consistent"],
                        "correct_option_wrong_reasoning": score["correct_option_wrong_reasoning"],
                        "prediction": completion,
                        "parsed_prediction": score["parsed"],
                    }
                )
    return rows


if __name__ == "__main__":
    main()
