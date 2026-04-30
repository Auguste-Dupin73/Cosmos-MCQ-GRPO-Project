from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer, BitsAndBytesConfig
from trl import GRPOConfig, GRPOTrainer

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.dataset import load_episode_records, to_hf_dataset
from training.reward_fn import build_reward_functions
from training.utils import (
    ProgressMetricsCallback,
    coerce_path_list,
    coerce_torch_dtype,
    latest_checkpoint,
    load_yaml_config,
    resolve_output_path,
    set_seed,
    split_records,
    take_limit,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a GRPO policy on episode-style math data.")
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    parser.add_argument("--resume_from_checkpoint", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--model_name_or_path", default=None)
    parser.add_argument("--train_path", action="append", default=None)
    parser.add_argument("--eval_path", action="append", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    config = apply_overrides(config, args)

    seed = int(config.get("seed", 42))
    set_seed(seed)

    model_cfg = dict(config.get("model", {}))
    data_cfg = dict(config.get("data", {}))
    training_cfg = dict(config.get("training", {}))
    peft_config = build_peft_config(config.get("peft"))

    model_name_or_path = model_cfg["name_or_path"]
    tokenizer = load_tokenizer(model_cfg)

    train_records, eval_records = prepare_datasets(data_cfg, seed=seed)
    print(
        json.dumps(
            {
                "train_examples": len(train_records),
                "eval_examples": len(eval_records),
                "dataset_format": train_records[0]["dataset_format"] if train_records else None,
                "model_name_or_path": model_name_or_path,
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    reward_funcs, reward_weights = build_reward_functions(config.get("reward"))
    grpo_args = build_grpo_config(
        training_cfg=training_cfg,
        model_cfg=model_cfg,
        reward_weights=reward_weights,
        output_dir_override=args.output_dir,
        do_eval=bool(eval_records),
    )

    callbacks = [ProgressMetricsCallback(num_generations=grpo_args.num_generations or 1)]
    trainer = GRPOTrainer(
        model=model_name_or_path,
        reward_funcs=reward_funcs,
        args=grpo_args,
        train_dataset=to_hf_dataset(train_records),
        eval_dataset=to_hf_dataset(eval_records) if eval_records else None,
        processing_class=tokenizer,
        callbacks=callbacks,
        peft_config=peft_config,
    )

    resume_checkpoint = args.resume_from_checkpoint
    if resume_checkpoint is None and bool(training_cfg.get("auto_resume", False)):
        latest = latest_checkpoint(grpo_args.output_dir)
        resume_checkpoint = str(latest) if latest is not None else None

    train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)
    trainer.save_model()
    tokenizer.save_pretrained(grpo_args.output_dir)
    trainer.save_state()
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)

    if eval_records and bool(training_cfg.get("run_final_eval", True)):
        eval_metrics = trainer.evaluate()
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)


def prepare_datasets(data_cfg: dict[str, Any], *, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    dataset_format = data_cfg.get("dataset_format", "auto")
    include_support_pack = bool(data_cfg.get("include_support_pack", True))
    append_response_format = bool(data_cfg.get("append_response_format", False))

    train_paths = coerce_path_list(data_cfg.get("train_paths") or data_cfg.get("train_path"))
    if not train_paths:
        raise ValueError("Config must define data.train_path or data.train_paths")

    base_train_records = load_episode_records(
        [str(path) for path in train_paths],
        dataset_format=dataset_format,
        include_support_pack=include_support_pack,
        append_response_format=append_response_format,
        max_samples=None,
        shuffle=bool(data_cfg.get("shuffle_train", True)),
        seed=seed,
    )

    eval_paths = coerce_path_list(data_cfg.get("eval_paths") or data_cfg.get("eval_path"))
    if eval_paths:
        train_records = take_limit(base_train_records, data_cfg.get("max_train_samples"))
        eval_records = load_episode_records(
            [str(path) for path in eval_paths],
            dataset_format=dataset_format,
            include_support_pack=include_support_pack,
            append_response_format=append_response_format,
            max_samples=data_cfg.get("max_eval_samples"),
            shuffle=False,
            seed=seed,
        )
        return train_records, eval_records

    eval_split = data_cfg.get("eval_split")
    train_records, eval_records = split_records(base_train_records, eval_split, seed=seed)
    train_records = take_limit(train_records, data_cfg.get("max_train_samples"))
    eval_records = take_limit(eval_records, data_cfg.get("max_eval_samples"))
    return train_records, eval_records


def load_tokenizer(model_cfg: dict[str, Any]):
    name_or_path = model_cfg["tokenizer_name_or_path"] if model_cfg.get("tokenizer_name_or_path") else model_cfg["name_or_path"]
    tokenizer = AutoTokenizer.from_pretrained(
        name_or_path,
        trust_remote_code=bool(model_cfg.get("trust_remote_code", False)),
        local_files_only=bool(model_cfg.get("local_files_only", False)),
    )
    tokenizer.truncation_side = "left"
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
        else:  # pragma: no cover
            raise ValueError("Tokenizer is missing pad_token, eos_token, and unk_token")
    return tokenizer


def build_grpo_config(
    *,
    training_cfg: dict[str, Any],
    model_cfg: dict[str, Any],
    reward_weights: list[float],
    output_dir_override: str | None,
    do_eval: bool,
) -> GRPOConfig:
    output_dir = resolve_output_path(output_dir_override or training_cfg["output_dir"])
    model_init_kwargs = dict(model_cfg.get("model_init_kwargs", {}))
    if "torch_dtype" in model_cfg:
        model_init_kwargs["torch_dtype"] = coerce_torch_dtype(model_cfg["torch_dtype"])
    if "trust_remote_code" in model_cfg:
        model_init_kwargs["trust_remote_code"] = bool(model_cfg["trust_remote_code"])
    if "local_files_only" in model_cfg:
        model_init_kwargs["local_files_only"] = bool(model_cfg["local_files_only"])
    if "quantization" in model_cfg:
        model_init_kwargs["quantization_config"] = build_quantization_config(model_cfg["quantization"])

    config_kwargs = {
        "output_dir": str(output_dir),
        "do_train": True,
        "do_eval": do_eval,
        "remove_unused_columns": False,
        "reward_weights": reward_weights,
        "model_init_kwargs": model_init_kwargs,
    }
    config_kwargs.update(training_cfg)
    config_kwargs.pop("auto_resume", None)
    config_kwargs.pop("run_final_eval", None)
    config_kwargs["output_dir"] = str(output_dir)
    config_kwargs["remove_unused_columns"] = False
    config_kwargs["reward_weights"] = reward_weights
    config_kwargs = filter_grpo_config_kwargs(config_kwargs)
    return GRPOConfig(**config_kwargs)


def build_quantization_config(raw_config: dict[str, Any]) -> BitsAndBytesConfig:
    quant_config = dict(raw_config or {})
    for key in ("bnb_4bit_compute_dtype", "bnb_4bit_quant_storage"):
        if key in quant_config:
            quant_config[key] = coerce_torch_dtype(quant_config[key])
    try:
        return BitsAndBytesConfig(**quant_config)
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "bitsandbytes is required for model.quantization. Install with: pip install bitsandbytes"
        ) from exc


def build_peft_config(raw_config: dict[str, Any] | None):
    raw_config = dict(raw_config or {})
    if not raw_config or not bool(raw_config.pop("enabled", False)):
        return None
    try:
        from peft import LoraConfig
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PEFT is required for peft.enabled=true. Install with: pip install peft") from exc
    return LoraConfig(**raw_config)


def filter_grpo_config_kwargs(config_kwargs: dict[str, Any]) -> dict[str, Any]:
    """Keep config files portable across nearby TRL/Transformers versions."""
    signature = inspect.signature(GRPOConfig.__init__)
    allowed = {name for name in signature.parameters if name != "self"}
    filtered = dict(config_kwargs)

    if "eval_strategy" in filtered and "eval_strategy" not in allowed and "evaluation_strategy" in allowed:
        filtered["evaluation_strategy"] = filtered.pop("eval_strategy")

    unsupported = sorted(key for key in filtered if key not in allowed)
    if unsupported:
        print(
            json.dumps(
                {
                    "warning": "Ignoring GRPOConfig keys unsupported by the installed TRL/Transformers version.",
                    "unsupported_keys": unsupported,
                },
                ensure_ascii=False,
                indent=2,
            ),
            file=sys.stderr,
        )
        for key in unsupported:
            filtered.pop(key, None)
    return filtered


def apply_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    config = dict(config)
    config.setdefault("model", {})
    config.setdefault("data", {})
    config.setdefault("training", {})

    if args.model_name_or_path:
        config["model"]["name_or_path"] = args.model_name_or_path
    if args.output_dir:
        config["training"]["output_dir"] = args.output_dir
    if args.train_path:
        config["data"]["train_paths"] = args.train_path
    if args.eval_path:
        config["data"]["eval_paths"] = args.eval_path
    return config


if __name__ == "__main__":
    main()
