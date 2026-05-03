# Training

This folder contains the GRPO-first training layer for the episode-style math dataset. It is isolated from the existing generation and conversion pipeline on purpose.

## What Is Here

- `train_grpo.py`: main GRPO training entrypoint built on `trl.GRPOTrainer`
- `eval_grpo.py`: structured checkpoint/model evaluation on episode data
- `reward_fn.py`: reward logic and diagnostic reward metrics
- `dataset.py`: loaders for source episodes plus converted offline/online episode GRPO files
- `formatting.py`: prompt rendering and completion parsing
- `utils.py`: config, seeding, JSONL, checkpoint, and metric helpers
- `configs/`: pilot and fuller YAML templates
- `scripts/`: thin launch helpers

## Required Dependencies

The existing repo README only mentions `pytest`, but the training layer expects the Hugging Face stack that is already referenced by `cli/SFT.py`:

```powershell
python -m pip install torch transformers datasets trl pyyaml accelerate peft bitsandbytes
```

Optional:

```powershell
python -m pip install wandb
```

## Recommended Input Files

Best training input:

- source episode JSONL such as `outputs/200_sample/episodes.jsonl`
- this keeps full `main`, `probe`, `support_pack`, and reasoning targets available to the reward function
- current Qwen configs use `split_main_probe: true`, so each source episode becomes two prompt-level training rows: one main MCQ prompt and one separate probe prompt

Also supported:

- `episode_grpo_offline.jsonl`
- `episode_grpo_online.jsonl`

Notes:

- source episode files give the strongest reasoning-aware reward shaping
- `train_grpo.py` should normally point to `episodes.jsonl`, not `episode_grpo_online.jsonl`
- `episode_grpo_online_split.jsonl` is available for inspection/export, but the trainer gets the richest reward metadata from source `episodes.jsonl`
- offline files are still useful because the loader can recover gold targets from the best candidate response
- online files work, but reasoning checks are necessarily weaker because they do not store the full intermediate-step structure

## Pilot Training

```powershell
python training/train_grpo.py --config training/configs/pilot.yaml
```

## Debug Raw Completions

Before a GRPO run, inspect raw generations with the same config:

```powershell
python training/debug_completions.py --config training/configs/colab_qwen4b.yaml --num-prompts 2 --num-generations 2
```

Useful Colab variant that saves JSONL diagnostics:

```powershell
python training/debug_completions.py `
  --config training/configs/colab_qwen4b.yaml `
  --num-prompts 4 `
  --num-generations 2 `
  --output outputs/training/debug_completions.jsonl
```

Look for `token_count`, `terminated`, `clipped`, the raw completion text, and parsed fields such as `main_selected_option`, `main_final_answer`, and `probe_final_answer`. If completions are empty after special-token stripping or terminate after one token, the model is emitting EOS immediately. If completions are clipped, the model is rambling past the configured generation cap.

## Fuller Training

```powershell
python training/train_grpo.py --config training/configs/full.yaml
```

## Evaluation

Evaluate a checkpoint on the held-out 500-episode test set:

```powershell
python training/eval_grpo.py `
  --config training/configs/eval_qwen4b_test500.yaml `
  --checkpoint path\to\checkpoint-or-model `
  --output outputs/training/eval_test500_report.json `
  --predictions_out outputs/training/eval_test500_predictions.jsonl
```

The test file has 500 source episodes and evaluates as 1000 prompt-level tasks because main and probe are tested separately.

Evaluate a checkpoint from the pilot run:

```powershell
python training/eval_grpo.py `
  --config training/configs/pilot.yaml `
  --checkpoint outputs/training/grpo_pilot/checkpoint-20
```

You can also override the eval data and save a report:

```powershell
python training/eval_grpo.py `
  --config training/configs/full.yaml `
  --checkpoint path\to\checkpoint-or-model `
  --data outputs/episodes_C.jsonl `
  --output outputs/training/eval_report.json `
  --predictions_out outputs/training/eval_predictions.jsonl
```

## Training Behavior

- `train_grpo.py` uses `trl.GRPOTrainer` directly
- `data.max_train_samples` controls how many prompt-level rows are loaded after splitting; if it is `8`, the run prints `train_examples: 8`
- `colab_qwen4b.yaml` and `full.yaml` now set `max_train_samples: null`, so the 200-episode training artifact loads as 400 prompt-level rows
- reward logic stays in `reward_fn.py`
- the trainer registers one real optimization reward and several zero-weight diagnostic rewards so training logs still expose:
  - reward mean/std
  - shaping reward
  - format compliance
  - main accuracy
  - option accuracy
  - probe accuracy
  - joint success
  - reasoning consistency
  - correct-option-wrong-reasoning rate
- checkpoints and metrics are saved under the configured `output_dir`

## Interpreting Zero Loss

GRPO can show `loss=0` when every completion in a sampled group receives the same reward. With a base model this usually means the model is not yet producing parseable fields such as `Ana soru secimi` or `Probe nihai cevap`, so all completions score zero and there is no advantage signal.

The reward module includes small capped shaping rewards for format compliance, answer-field presence, numeric work, and mentioning the gold answer. The correctness gates still dominate: full credit requires correct main answer, correct option, consistent reasoning, and correct probe answer.

## Known Limitations

- this repo snapshot does not include a local model checkpoint, so you need either a cached Hugging Face model or network access when loading `model.name_or_path`
- the current environment where this stack was added does not expose CUDA, so end-to-end runtime validation with a real model was limited to static and module-level checks
- live per-skill and per-tier breakdowns are provided by `eval_grpo.py`; they are not added to every in-training log event
- the Colab/L4 Qwen config uses 4-bit QLoRA; install `peft` and `bitsandbytes` before running it
- if an L4 run still OOMs, lower `max_completion_length`, keep `include_support_pack: false`, reduce `num_generations`, or use the pilot config before the full config
