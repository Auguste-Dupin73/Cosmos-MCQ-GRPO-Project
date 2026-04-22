# Cosmos Math Data Generation

CLI manual for the math data generation workflows in this repository.

This repo currently contains two pipelines:

- `legacy single-sample pipeline`: generate raw text questions, then convert them into SFT/DPO/GRPO datasets
- `episode-style GRPO pipeline`: build structured MCQ episodes with adversarial candidates, then convert them into episode SFT/DPO/GRPO datasets

All commands below should be run from the repository root:

```powershell
cd c:\Users\Acer\Desktop\Cosmos-Proje-Grpo
```

## Requirements

- Python 3
- `pytest` only if you want to run tests

Install pytest if needed:

```powershell
python -m pip install pytest
```

## Repository Layout

- `cli/`: CLI entry points for building, converting, validating, and training helper scripts
- `example-generator-v1/`: legacy raw question generator and seed utilities
- `raw-to-sft-grpo-dpo-formatter/`: legacy dataset converter
- `episode_builder.py`, `mcq_builder.py`, `adversarial_negatives.py`, `mcq_consistency.py`: shared episode pipeline modules
- `cli/build_grpo_episodes.py`: build source episode JSONL files
- `cli/convert_episodes.py`: convert episode JSONL into SFT/DPO/GRPO datasets
- `cli/validate_episode_jsonl.py`: validate source episode files
- `cli/validate_episode_grpo_offline.py`: validate converted offline GRPO episode files
- `outputs/`: generated episode files and converted outputs
- `tests/`: pytest coverage for the episode pipeline

## Path Resolution

The episode scripts accept paths relative to any of these locations:

- repo root
- `example-generator-v1/`
- `example-generator-v1/data/`
- `raw-to-sft-grpo-dpo-formatter/`

That means commands like `--seeds seeds_llm_clean_A.jsonl` work from the repo root.

## Episode Pipeline

### 1. Build Source Episodes

Tier A:

```powershell
python cli/build_grpo_episodes.py `
  --seeds seeds_llm_clean_A.jsonl `
  --phrase_bank phrase_bank_tierA_all.json `
  --out outputs/episodes_A.jsonl `
  --variants_per_seed 1 `
  --seed 42
```

Tier B:

```powershell
python cli/build_grpo_episodes.py `
  --seeds seeds_llm_clean_B.jsonl `
  --phrase_bank phrase_bank_tierB_all.json `
  --out outputs/episodes_B.jsonl `
  --variants_per_seed 1 `
  --seed 42
```

Tier C:

```powershell
python cli/build_grpo_episodes.py `
  --seeds seeds_llm_clean_C.jsonl `
  --phrase_bank phrase_bank_tierC_all.json `
  --out outputs/episodes_C.jsonl `
  --variants_per_seed 1 `
  --seed 42
```

What this writes:

- `outputs/episodes_A.jsonl`
- `outputs/episodes_B.jsonl`
- `outputs/episodes_C.jsonl`

Each row is a full episode record with:

- `main`
- `probe`
- `support_pack`
- `reward_spec`
- `adversarial_candidates`

### 2. Validate Source Episodes

Validate a source episode file:

```powershell
python cli/validate_episode_jsonl.py --input outputs/episodes_A.jsonl
```

You can do the same for B and C:

```powershell
python cli/validate_episode_jsonl.py --input outputs/episodes_B.jsonl
python cli/validate_episode_jsonl.py --input outputs/episodes_C.jsonl
```

The validator reports:

- total episodes
- total adversarial candidates
- mismatch count
- unmappable candidate count
- duplicate option count
- sample failures

Exit code:

- `0` means valid
- non-zero means at least one validation failure was found

### 3. Convert Episodes Into Training Files

Convert Tier A:

```powershell
python cli/convert_episodes.py `
  --input outputs/episodes_A.jsonl `
  --out_dir outputs/converted_A
```

Convert Tier B:

```powershell
python cli/convert_episodes.py `
  --input outputs/episodes_B.jsonl `
  --out_dir outputs/converted_B
```

Convert Tier C:

```powershell
python cli/convert_episodes.py `
  --input outputs/episodes_C.jsonl `
  --out_dir outputs/converted_C
```

Each converted directory contains:

- `episode_sft.jsonl`
- `episode_dpo.jsonl`
- `episode_grpo_offline.jsonl`
- `episode_grpo_online.jsonl`

### 4. Validate Offline GRPO Outputs

Validate converted offline data:

```powershell
python cli/validate_episode_grpo_offline.py --input outputs/converted_A/episode_grpo_offline.jsonl
```

Also valid for B and C:

```powershell
python cli/validate_episode_grpo_offline.py --input outputs/converted_B/episode_grpo_offline.jsonl
python cli/validate_episode_grpo_offline.py --input outputs/converted_C/episode_grpo_offline.jsonl
```

This checks that each serialized response is MCQ-consistent with the prompt options.

## Quick Start

If you only want the new episode pipeline for Tier A:

```powershell
python cli/build_grpo_episodes.py --seeds seeds_llm_clean_A.jsonl --phrase_bank phrase_bank_tierA_all.json --out outputs/episodes_A.jsonl --variants_per_seed 1 --seed 42
python cli/validate_episode_jsonl.py --input outputs/episodes_A.jsonl
python cli/convert_episodes.py --input outputs/episodes_A.jsonl --out_dir outputs/converted_A
python cli/validate_episode_grpo_offline.py --input outputs/converted_A/episode_grpo_offline.jsonl
```

## Legacy Pipeline

The legacy path is still available and unchanged.

### 1. Generate Raw Questions

Example Tier A command:

```powershell
python example-generator-v1\generate_raw.py `
  --seeds example-generator-v1/data/team_seeds/tierA/codexA_shuffled.jsonl `
  --phrase_bank example-generator-v1/data/phrase_bank_tierA_all.json `
  --out example-generator-v1/data/out/shuffle_A.txt `
  --variants_per_seed 3 `
  --seed 42 `
  --dump_stats
```

### 2. Convert Raw Questions

```powershell
python raw-to-sft-grpo-dpo-formatter\convert.py `
  --input example-generator-v1/data/out/shuffle_A.txt `
  --input_type txt `
  --out_dir raw-to-sft-grpo-dpo-formatter/data/shuffleA `
  --seed 42 `
  --id_prefix gen_A
```

For more legacy examples, see:

- [example-generator-v1/Commands.txt](/c:/Users/Acer/Desktop/Cosmos-Proje-Grpo/example-generator-v1/Commands.txt)

## Tests

Run the main episode-pipeline test set:

```powershell
python -m pytest tests/test_mcq_consistency.py tests/test_episode_builder.py tests/test_convert_episodes.py tests/test_reasoning_checks.py tests/test_mcq_builder.py -q
```

If your environment has temp-directory permission issues, use:

```powershell
python -m pytest tests/test_mcq_consistency.py tests/test_episode_builder.py tests/test_convert_episodes.py tests/test_reasoning_checks.py tests/test_mcq_builder.py -p no:tmpdir -p no:cacheprovider -q
```

## Output Files

Common output locations:

- source episodes: `outputs/episodes_A.jsonl`, `outputs/episodes_B.jsonl`, `outputs/episodes_C.jsonl`
- converted episode datasets: `outputs/converted_A/`, `outputs/converted_B/`, `outputs/converted_C/`
- legacy formatted datasets: `raw-to-sft-grpo-dpo-formatter/data/`

## Troubleshooting

`Could not resolve path`

- Run commands from the repo root
- Check the file name
- Use a full relative path if needed

`Validation script exits non-zero`

- Read the JSON summary it prints
- Fix the offending file or rebuild it with the episode pipeline

`pytest` not found

- Install it with `python -m pip install pytest`
