# Cosmos Data Pipeline Tutorial

This repository has two dataset pipelines:

1. Legacy single-sample pipeline
2. Episode-style GRPO pipeline

Use the episode-style pipeline for current GRPO work. Use the legacy pipeline only when you need the older raw-question SFT/DPO/GRPO format.

All commands below assume you are at the repository root:

```powershell
cd C:\Users\Acer\Desktop\Metin\Cosmos\Cosmos-Proje-Grpo
```

## Outputs Created In This Run

The 200-row episode artifact is saved here:

```text
outputs/200_sample/
```

It contains:

```text
outputs/200_sample/episodes.jsonl
outputs/200_sample/episode_sft.jsonl
outputs/200_sample/episode_sft_split.jsonl
outputs/200_sample/episode_dpo.jsonl
outputs/200_sample/episode_grpo_offline.jsonl
outputs/200_sample/episode_grpo_offline_split.jsonl
outputs/200_sample/episode_grpo_online.jsonl
outputs/200_sample/episode_grpo_online_split.jsonl
outputs/200_sample/manifest.json
outputs/200_sample/selected_seeds/tierA_selected_50.jsonl
outputs/200_sample/selected_seeds/tierB_selected_30.jsonl
outputs/200_sample/selected_seeds/tierC_selected_20.jsonl
outputs/200_sample/tierA_episodes.jsonl
outputs/200_sample/tierB_episodes.jsonl
outputs/200_sample/tierC_episodes.jsonl
```

The requested mix is:

```text
Tier A: 50 seeds x 2 variants = 100 episodes
Tier B: 30 seeds x 2 variants = 60 episodes
Tier C: 20 seeds x 2 variants = 40 episodes
Total: 100 seeds x 2 variants = 200 episodes
```

The manifest records the sampling plan, selected seed files, per-tier counts, template mix, skipped invalid probe candidates, and final combined episode file.

## Episode-Style GRPO Pipeline

This is the preferred pipeline for GRPO training.

### What A Source Episode Contains

Each row in `episodes.jsonl` is a structured episode with:

```text
episode_id
skill_id
template_id
tier
difficulty
language
main
probe
support_pack
reward_spec
adversarial_candidates
tags
```

The important training pieces are:

```text
main: main MCQ question, options, gold option, gold final answer, solution, intermediates
probe: same-skill follow-up MCQ question, options, gold option, gold answer
support_pack: skill hints and mini example
reward_spec: gated reward requirements
adversarial_candidates: wrong or partially correct candidate responses
```

### Build Source Episodes

For a normal per-tier build:

```powershell
python cli/build_grpo_episodes.py `
  --seeds example-generator-v1/data/team_seeds/tierA/codexA_shuffled.jsonl `
  --phrase_bank phrase_bank_tierA_all.json `
  --out outputs/episodes_A.jsonl `
  --variants_per_seed 1 `
  --seed 42
```

Tier B:

```powershell
python cli/build_grpo_episodes.py `
  --seeds example-generator-v1/data/team_seeds/tierB/codex_B.jsonl `
  --phrase_bank phrase_bank_tierB_all.json `
  --out outputs/episodes_B.jsonl `
  --variants_per_seed 1 `
  --seed 42
```

Tier C:

```powershell
python cli/build_grpo_episodes.py `
  --seeds example-generator-v1/data/team_seeds/tierC/codex_C.jsonl `
  --phrase_bank phrase_bank_tierC_all.json `
  --out outputs/episodes_C.jsonl `
  --variants_per_seed 1 `
  --seed 42
```

For the 200-sample artifact, selected seed files were built first:

```text
outputs/200_sample/selected_seeds/tierA_selected_50.jsonl
outputs/200_sample/selected_seeds/tierB_selected_30.jsonl
outputs/200_sample/selected_seeds/tierC_selected_20.jsonl
```

Then each tier was rendered with `variants_per_seed=2`, producing:

```text
outputs/200_sample/tierA_episodes.jsonl
outputs/200_sample/tierB_episodes.jsonl
outputs/200_sample/tierC_episodes.jsonl
```

Those files were merged into:

```text
outputs/200_sample/episodes.jsonl
```

Episode IDs were rewritten during merge so the combined file has unique IDs.

### Validate Source Episodes

Validate a source episode file:

```powershell
python cli/validate_episode_jsonl.py --input outputs/200_sample/episodes.jsonl
```

Expected success shape:

```json
{
  "total_episodes": 200,
  "total_adversarial_candidates": 1200,
  "mismatch_count": 0,
  "unmappable_candidate_count": 0,
  "duplicate_option_count": 0,
  "family_failure_breakdown": {},
  "samples": []
}
```

### Convert Source Episodes

Convert source episodes into episode training files:

```powershell
python cli/convert_episodes.py `
  --input outputs/200_sample/episodes.jsonl `
  --out_dir outputs/200_sample
```

This writes:

```text
episode_sft.jsonl
episode_dpo.jsonl
episode_grpo_offline.jsonl
episode_grpo_online.jsonl
```

### Validate Offline GRPO Output

```powershell
python cli/validate_episode_grpo_offline.py `
  --input outputs/200_sample/episode_grpo_offline.jsonl
```

Expected success shape:

```json
{
  "total_candidates_scanned": 1400,
  "mismatch_count": 0,
  "unmappable_final_answer_count": 0,
  "duplicate_option_count": 0,
  "samples": []
}
```

### Episode Output Formats

`episode_sft.jsonl`

One gold answer per prompt.

```json
{
  "id": "...",
  "prompt": "...",
  "answer": "...",
  "tags": ["..."]
}
```

`episode_dpo.jsonl`

One chosen response and one rejected response.

```json
{
  "id": "...",
  "prompt": "...",
  "chosen": "...",
  "rejected": "...",
  "tags": ["..."]
}
```

`episode_grpo_offline.jsonl`

One prompt with several scored candidate responses.

```json
{
  "id": "...",
  "prompt": "...",
  "responses": [
    {"text": "...", "score": 1.0},
    {"text": "...", "score": 0.0}
  ],
  "reward_spec": {"scoring_mode": "gated"},
  "tags": ["..."]
}
```

`episode_grpo_online.jsonl`

One prompt with gold metadata for online reward computation.

```json
{
  "id": "...",
  "prompt": "...",
  "gold": {
    "main_gold_option": "A",
    "main_gold_final_answer": "108",
    "probe_gold_final_answer": "162"
  },
  "reward_spec": {"scoring_mode": "gated"},
  "tags": ["..."]
}
```

`episode_grpo_online_split.jsonl`

Main and probe are separate prompt rows. This is the preferred export shape for Qwen debugging and eval:

```json
{
  "id": "...::main",
  "episode_id": "...",
  "task_type": "main",
  "prompt": "Matematik sorusunu coz...\n<final>\noption:\nmain:\n</final>",
  "gold": {
    "main_gold_option": "A",
    "main_gold_final_answer": "108",
    "probe_gold_final_answer": "162"
  }
}
```

Probe split rows use the same MCQ response shape:

```json
{
  "id": "...::probe",
  "episode_id": "...",
  "task_type": "probe",
  "prompt": "Matematik sorusunu coz...\nSecenekler:\nA) ...\nB) ...\nC) ...\nD) ...\n<final>\noption:\nprobe:\n</final>",
  "gold": {
    "probe_gold_option": "C",
    "probe_gold_final_answer": "162"
  }
}
```

The source `episodes.jsonl` remains the recommended training input because the trainer can split it dynamically while retaining all reward metadata.

## Legacy Single-Sample Pipeline

The legacy pipeline creates plain math questions first, then converts them into older SFT/DPO/GRPO files.

Use this pipeline when you want the old single-question format. Do not use it for the episode-style GRPO training stack unless you intentionally need legacy data.

### Generate Raw Questions

Example Tier A generation:

```powershell
python example-generator-v1/generate_raw.py `
  --seeds example-generator-v1/data/team_seeds/tierA/codexA_shuffled.jsonl `
  --phrase_bank example-generator-v1/data/phrase_bank_tierA_all.json `
  --out example-generator-v1/data/out/tierA_raw.txt `
  --variants_per_seed 2 `
  --seed 42 `
  --dump_stats
```

Tier B and Tier C use their own seed files and phrase banks:

```text
example-generator-v1/data/team_seeds/tierB/codex_B.jsonl
example-generator-v1/data/phrase_bank_tierB_all.json

example-generator-v1/data/team_seeds/tierC/codex_C.jsonl
example-generator-v1/data/phrase_bank_tierC_all.json
```

### Convert Legacy Raw Questions

```powershell
python raw-to-sft-grpo-dpo-formatter/convert.py `
  --input example-generator-v1/data/out/tierA_raw.txt `
  --input_type txt `
  --out_dir raw-to-sft-grpo-dpo-formatter/data/tierA_converted `
  --seed 42 `
  --id_prefix tierA
```

The legacy converter writes:

```text
sft.jsonl
dpo.jsonl
grpo_offline.jsonl
grpo_online.jsonl
```

### Legacy Output Formats

Legacy SFT:

```json
{"id": "ex_000001", "prompt": "...", "answer": "..."}
```

Legacy DPO:

```json
{"id": "ex_000001", "prompt": "...", "chosen": "...", "rejected": "..."}
```

Legacy GRPO offline:

```json
{
  "id": "ex_000001",
  "prompt": "...",
  "responses": [
    {"text": "...", "score": 1.0},
    {"text": "...", "score": 0.0}
  ]
}
```

Legacy GRPO online:

```json
{
  "id": "ex_000001",
  "prompt": "...",
  "gold": "..."
}
```

## Choosing The Right Pipeline

Use episode-style when:

- training GRPO on main-question plus probe behavior
- using reward specs
- needing MCQ option consistency
- needing adversarial candidate families
- using `training/train_grpo.py`

Use legacy when:

- creating plain single-question datasets
- reproducing older SFT/DPO/GRPO experiments
- using raw text input from `example-generator-v1/generate_raw.py`

## Rebuilding The 200-Sample Artifact

The current 200-sample artifact was created by:

1. Deduplicating seed rows inside each tier directory.
2. Sampling valid buildable seeds with template diversity.
3. Keeping 50 A, 30 B, and 20 C seeds.
4. Rendering two episode variants per selected seed.
5. Merging per-tier source episodes into `outputs/200_sample/episodes.jsonl`.
6. Converting the merged file with `cli/convert_episodes.py`.
7. Validating source and offline GRPO outputs.

The exact selected seeds are stored in:

```text
outputs/200_sample/selected_seeds/
```

Use those selected seed files if you want to rebuild the same sample deterministically.

## Training With The Episode Files

For GRPO training, point the training config at the source episode file:

```yaml
data:
  dataset_format: source_episode
  split_main_probe: true
  train_paths:
    - outputs/200_sample/episodes.jsonl
```

With `split_main_probe: true`, each source episode becomes:

```text
1 main MCQ prompt
1 separate probe MCQ prompt
```

So `outputs/200_sample/episodes.jsonl` has 200 source episodes, but the trainer sees 400 prompt-level examples when `max_train_samples: null`.

If you see `train_examples: 8`, the cause is not the dataset size. It means the active YAML has:

```yaml
data:
  max_train_samples: 8
```

Remove the cap or set it to `null` to load all split prompt rows.

Then run:

```powershell
python training/train_grpo.py --config training/configs/colab_qwen4b.yaml
```

For standalone evaluation:

```powershell
python training/eval_grpo.py `
  --config training/configs/eval_qwen4b_test500.yaml `
  --checkpoint outputs/training/grpo_pilot/checkpoint-20 `
  --output outputs/training/eval_test500_report.json `
  --predictions_out outputs/training/eval_test500_predictions.jsonl
```

The held-out test dataset is:

```text
training/test_500/episodes.jsonl
```

It contains 500 source episodes with a 50% A, 30% B, 20% C tier mix and no exact main/probe question-text overlap with `outputs/200_sample/episodes.jsonl`. With split eval enabled, it becomes 1000 prompt-level eval tasks.

The eval report includes overall `option_accuracy`, plus separate `main_option_accuracy` and `probe_option_accuracy`.
