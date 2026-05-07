from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

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
    if not getattr(model, "is_loaded_in_4bit", False) and not getattr(model, "hf_device_map", None):
        model.to(device)
    model.eval()

    batch_size = int(eval_cfg.get("batch_size", 1))
    max_new_tokens = int(eval_cfg.get("max_new_tokens", 192))
    progress_interval = int(eval_cfg.get("progress_interval", 10))
    flush_interval = int(eval_cfg.get("flush_interval", 25))
    use_chat_template = bool(model_cfg.get("use_chat_template", False))
    predictions_path = resolve_output_path(args.predictions_out) if args.predictions_out else None
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
        predictions_path=predictions_path,
        progress_interval=progress_interval,
        flush_interval=flush_interval,
        use_chat_template=use_chat_template,
    )
    report = {
        "model_name_or_path": str(model_name_or_path),
        "model_input_format": "chat_template" if use_chat_template else "raw_prompt",
        "num_examples": len(rows),
        "overall": summarize_metric_rows(rows),
        "by_skill": summarize_metric_groups(rows, "skill_id"),
        "by_tier": summarize_metric_groups(rows, "tier"),
        "by_template": summarize_metric_groups(rows, "template_id"),
        "by_task_type": summarize_metric_groups(rows, "task_type"),
    }

    print(json.dumps(report, ensure_ascii=False, indent=2))

    if args.output:
        output_path = resolve_output_path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if predictions_path:
        write_jsonl(predictions_path, rows)


def load_eval_records(data_cfg: dict[str, Any], *, override_paths: list[str] | None) -> list[dict[str, Any]]:
    dataset_format = data_cfg.get("dataset_format", "auto")
    include_support_pack = bool(data_cfg.get("include_support_pack", True))
    append_response_format = bool(data_cfg.get("append_response_format", False))
    split_main_probe = bool(data_cfg.get("split_main_probe", False))

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
        split_main_probe=split_main_probe,
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
    if "quantization" in model_cfg:
        model_kwargs["quantization_config"] = build_quantization_config(dict(model_cfg["quantization"]))
        if torch.cuda.is_available():
            model_kwargs.setdefault("device_map", "auto")

    adapter_path = Path(model_name_or_path)
    if adapter_path.exists() and (adapter_path / "adapter_config.json").exists():
        base = AutoModelForCausalLM.from_pretrained(model_cfg["name_or_path"], **model_kwargs)
        try:
            from peft import PeftModel
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("PEFT is required to evaluate a LoRA adapter checkpoint. Install with: pip install peft") from exc
        return PeftModel.from_pretrained(base, str(adapter_path))
    return AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)


def build_quantization_config(raw_config: dict[str, Any]) -> BitsAndBytesConfig:
    for key in ("bnb_4bit_compute_dtype", "bnb_4bit_quant_storage"):
        if key in raw_config:
            raw_config[key] = coerce_torch_dtype(raw_config[key])
    try:
        return BitsAndBytesConfig(**raw_config)
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("bitsandbytes is required for model.quantization. Install with: pip install bitsandbytes") from exc


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
    predictions_path: Path | None = None,
    progress_interval: int = 10,
    flush_interval: int = 25,
    use_chat_template: bool = False,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    total = len(records)
    started_at = time.monotonic()
    next_progress_at = progress_interval if progress_interval > 0 else total
    last_flush_count = 0
    print(
        f"[eval] Starting {total} records with batch_size={batch_size}, "
        f"max_new_tokens={max_new_tokens}, "
        f"input_format={'chat_template' if use_chat_template else 'raw_prompt'}",
        file=sys.stderr,
        flush=True,
    )
    with torch.no_grad():
        for start in range(0, len(records), batch_size):
            batch = records[start : start + batch_size]
            prompts = [record["prompt"] for record in batch]
            model_prompts = apply_chat_template(tokenizer, prompts) if use_chat_template else prompts
            encoded = tokenizer(model_prompts, padding=True, return_tensors="pt")
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
                    task_type=record.get("task_type"),
                    reward_config=reward_cfg,
                )
                task_type = record.get("task_type", "episode")
                rows.append(
                    {
                        "id": record["id"],
                        "episode_id": record.get("episode_id"),
                        "task_type": task_type,
                        "skill_id": record.get("skill_id"),
                        "template_id": record.get("template_id"),
                        "tier": record.get("tier"),
                        "reward": score["reward"],
                        "main_accuracy": score["main_accuracy"] if task_type != "probe" else None,
                        "option_accuracy": score["option_accuracy"],
                        "main_option_accuracy": score["main_option_accuracy"] if task_type != "probe" else None,
                        "probe_option_accuracy": score["probe_option_accuracy"] if task_type != "main" else None,
                        "probe_accuracy": score["probe_accuracy"] if task_type != "main" else None,
                        "joint_success": score["joint_success"],
                        "reasoning_consistent": score["reasoning_consistent"],
                        "correct_option_wrong_reasoning": score["correct_option_wrong_reasoning"],
                        "prediction": completion,
                        "parsed_prediction": score["parsed"],
                    }
                )
            completed = len(rows)
            if completed >= next_progress_at or completed == total:
                _print_progress(completed, total, started_at)
                while progress_interval > 0 and next_progress_at <= completed:
                    next_progress_at += progress_interval
            if predictions_path and flush_interval > 0 and completed - last_flush_count >= flush_interval:
                write_jsonl(predictions_path, rows)
                last_flush_count = completed
                print(
                    f"[eval] Flushed {completed}/{total} predictions to {predictions_path}",
                    file=sys.stderr,
                    flush=True,
                )
    return rows


def apply_chat_template(tokenizer, prompts: list[str]) -> list[str]:
    try:
        return [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for prompt in prompts
        ]
    except Exception as exc:
        raise RuntimeError(
            "model.use_chat_template=true requires a tokenizer with a valid chat_template."
        ) from exc


def _print_progress(completed: int, total: int, started_at: float) -> None:
    elapsed = time.monotonic() - started_at
    rate = completed / elapsed if elapsed > 0 else 0.0
    remaining = (total - completed) / rate if rate > 0 else None
    percent = (completed / total * 100.0) if total else 100.0
    print(
        "[eval] "
        f"{completed}/{total} ({percent:.1f}%) "
        f"elapsed={_format_duration(elapsed)} "
        f"eta={_format_duration(remaining)} "
        f"rate={rate:.3f} examples/s",
        file=sys.stderr,
        flush=True,
    )


def _format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


if __name__ == "__main__":
    main()
