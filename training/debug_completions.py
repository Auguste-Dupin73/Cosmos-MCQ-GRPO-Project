from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.dataset import load_episode_records
from training.reward_fn import score_completion_against_episode
from training.utils import coerce_path_list, coerce_torch_dtype, load_yaml_config, set_seed, take_limit, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print raw completions and reward diagnostics for GRPO debugging.")
    parser.add_argument("--config", required=True, help="Path to a training YAML config.")
    parser.add_argument("--checkpoint", default=None, help="Optional model/checkpoint/LoRA adapter path.")
    parser.add_argument("--data", action="append", default=None, help="Optional override data path(s).")
    parser.add_argument("--num-prompts", type=int, default=2, help="Number of prompts to sample.")
    parser.add_argument("--num-generations", type=int, default=None, help="Completions per prompt.")
    parser.add_argument("--max-new-tokens", type=int, default=None, help="Generation length override.")
    parser.add_argument("--temperature", type=float, default=None, help="Sampling temperature override.")
    parser.add_argument("--top-p", type=float, default=None, help="Top-p override.")
    parser.add_argument("--top-k", type=int, default=None, help="Top-k override.")
    parser.add_argument("--greedy", action="store_true", help="Disable sampling.")
    parser.add_argument("--show-prompt", action="store_true", help="Print full prompts before completions.")
    parser.add_argument("--output", default=None, help="Optional JSONL output path for debug rows.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    seed = int(config.get("seed", 42))
    set_seed(seed)

    model_cfg = dict(config.get("model", {}))
    data_cfg = dict(config.get("data", {}))
    training_cfg = dict(config.get("training", {}))
    reward_cfg = dict(config.get("reward", {}))

    records = load_debug_records(data_cfg, override_paths=args.data, seed=seed, limit=args.num_prompts)
    if not records:
        raise ValueError("No debug records were loaded")

    tokenizer = load_tokenizer(model_cfg, args.checkpoint)
    model = load_model(model_cfg, args.checkpoint)
    model.eval()

    generation_cfg = build_generation_config(args, training_cfg, tokenizer)
    device = infer_input_device(model)
    rows: list[dict[str, Any]] = []

    print(
        json.dumps(
            {
                "model": args.checkpoint or model_cfg["name_or_path"],
                "prompts": len(records),
                "num_generations": generation_cfg["num_return_sequences"],
                "max_new_tokens": generation_cfg["max_new_tokens"],
                "do_sample": generation_cfg["do_sample"],
                "temperature": generation_cfg.get("temperature"),
                "top_p": generation_cfg.get("top_p"),
                "top_k": generation_cfg.get("top_k"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    with torch.no_grad():
        for prompt_index, record in enumerate(records, start=1):
            print_prompt_header(record, prompt_index, show_prompt=args.show_prompt)
            encoded = tokenizer(record["prompt"], return_tensors="pt")
            encoded = {key: value.to(device) for key, value in encoded.items()}
            generated = model.generate(**encoded, **generation_cfg)
            prompt_len = encoded["input_ids"].shape[1]

            for generation_index, output_ids in enumerate(generated, start=1):
                new_ids = output_ids[prompt_len:].detach().cpu().tolist()
                completion = tokenizer.decode(new_ids, skip_special_tokens=True)
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
                row = build_debug_row(
                    record=record,
                    prompt_index=prompt_index,
                    generation_index=generation_index,
                    completion=completion,
                    new_token_ids=new_ids,
                    tokenizer=tokenizer,
                    max_new_tokens=int(generation_cfg["max_new_tokens"]),
                    score=score,
                )
                rows.append(row)
                print_debug_row(row)

    if args.output:
        write_jsonl(args.output, rows)


def load_debug_records(
    data_cfg: Mapping[str, Any],
    *,
    override_paths: list[str] | None,
    seed: int,
    limit: int,
) -> list[dict[str, Any]]:
    paths = override_paths
    if not paths:
        paths = [str(path) for path in coerce_path_list(data_cfg.get("train_paths") or data_cfg.get("train_path"))]
    if not paths:
        raise ValueError("Config must define data.train_path or data.train_paths, or pass --data")

    records = load_episode_records(
        paths,
        dataset_format=data_cfg.get("dataset_format", "auto"),
        include_support_pack=bool(data_cfg.get("include_support_pack", True)),
        append_response_format=bool(data_cfg.get("append_response_format", False)),
        split_main_probe=bool(data_cfg.get("split_main_probe", False)),
        max_samples=None,
        shuffle=bool(data_cfg.get("shuffle_train", False)),
        seed=seed,
    )
    return take_limit(records, limit)


def load_tokenizer(model_cfg: Mapping[str, Any], checkpoint: str | None):
    tokenizer_name = model_cfg.get("tokenizer_name_or_path") or checkpoint or model_cfg["name_or_path"]
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
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


def load_model(model_cfg: Mapping[str, Any], checkpoint: str | None):
    model_kwargs = build_model_kwargs(model_cfg)
    adapter_path = Path(checkpoint) if checkpoint and Path(checkpoint).exists() else None
    if adapter_path is not None and (adapter_path / "adapter_config.json").exists():
        base = AutoModelForCausalLM.from_pretrained(model_cfg["name_or_path"], **model_kwargs)
        try:
            from peft import PeftModel
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("PEFT is required to debug a LoRA adapter checkpoint. Install with: pip install peft") from exc
        return PeftModel.from_pretrained(base, str(adapter_path))

    model_source = checkpoint or model_cfg["name_or_path"]
    model = AutoModelForCausalLM.from_pretrained(model_source, **model_kwargs)
    if "device_map" not in model_kwargs and torch.cuda.is_available():
        model.to("cuda")
    return model


def build_model_kwargs(model_cfg: Mapping[str, Any]) -> dict[str, Any]:
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
    return model_kwargs


def build_quantization_config(raw_config: dict[str, Any]) -> BitsAndBytesConfig:
    for key in ("bnb_4bit_compute_dtype", "bnb_4bit_quant_storage"):
        if key in raw_config:
            raw_config[key] = coerce_torch_dtype(raw_config[key])
    try:
        return BitsAndBytesConfig(**raw_config)
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("bitsandbytes is required for model.quantization. Install with: pip install bitsandbytes") from exc


def build_generation_config(args: argparse.Namespace, training_cfg: Mapping[str, Any], tokenizer) -> dict[str, Any]:
    do_sample = not args.greedy
    generation_cfg: dict[str, Any] = {
        "max_new_tokens": int(args.max_new_tokens or training_cfg.get("max_completion_length", 128)),
        "num_return_sequences": int(args.num_generations or training_cfg.get("num_generations", 2)),
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        generation_cfg["temperature"] = float(args.temperature or training_cfg.get("temperature", 0.8))
        generation_cfg["top_p"] = float(args.top_p or training_cfg.get("top_p", 1.0))
        generation_cfg["top_k"] = int(args.top_k if args.top_k is not None else training_cfg.get("top_k", 0))
    return generation_cfg


def infer_input_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:  # pragma: no cover
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_debug_row(
    *,
    record: Mapping[str, Any],
    prompt_index: int,
    generation_index: int,
    completion: str,
    new_token_ids: list[int],
    tokenizer,
    max_new_tokens: int,
    score: Mapping[str, Any],
) -> dict[str, Any]:
    eos_token_id = tokenizer.eos_token_id
    terminated = eos_token_id in new_token_ids if eos_token_id is not None else False
    token_count = len(new_token_ids)
    return {
        "prompt_index": prompt_index,
        "generation_index": generation_index,
        "id": record.get("id"),
        "episode_id": record.get("episode_id"),
        "task_type": record.get("task_type"),
        "skill_id": record.get("skill_id"),
        "template_id": record.get("template_id"),
        "tier": record.get("tier"),
        "token_count": token_count,
        "terminated": terminated,
        "clipped": token_count >= max_new_tokens and not terminated,
        "new_token_ids": new_token_ids if token_count <= 5 else None,
        "reward": score["reward"],
        "shaping_reward": score.get("shaping_reward"),
        "format_compliance": score.get("format_compliance"),
        "main_accuracy": score["main_accuracy"],
        "option_accuracy": score["option_accuracy"],
        "main_option_accuracy": score.get("main_option_accuracy"),
        "probe_option_accuracy": score.get("probe_option_accuracy"),
        "probe_accuracy": score["probe_accuracy"],
        "joint_success": score["joint_success"],
        "option_present": score.get("option_present"),
        "main_final_present": score.get("main_final_present"),
        "probe_final_present": score.get("probe_final_present"),
        "completion": completion,
        "parsed": score["parsed"],
    }


def print_prompt_header(record: Mapping[str, Any], prompt_index: int, *, show_prompt: bool) -> None:
    print("\n" + "=" * 88)
    print(
        f"Prompt {prompt_index}: id={record.get('id')} tier={record.get('tier')} "
        f"skill={record.get('skill_id')} template={record.get('template_id')} task={record.get('task_type')}"
    )
    if show_prompt:
        print("-" * 88)
        print(record["prompt"])


def print_debug_row(row: Mapping[str, Any]) -> None:
    print("-" * 88)
    print(
        "Completion {generation_index}: tokens={token_count} terminated={terminated} clipped={clipped} "
        "reward={reward:.4g} format={format_compliance:.4g} main={main_accuracy:.4g} "
        "option={option_accuracy:.4g} probe={probe_accuracy:.4g}".format(**row)
    )
    if row.get("new_token_ids") is not None:
        print(f"new_token_ids={row['new_token_ids']}")
    print("parsed=" + json.dumps(row["parsed"], ensure_ascii=False))
    print("raw_completion:")
    print(row["completion"] if row["completion"].strip() else "<EMPTY AFTER SPECIAL TOKENS>")


if __name__ == "__main__":
    main()
