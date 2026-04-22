from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

from mcq_consistency import extract_main_final_answer, extract_main_selected_option, resolve_option_for_value, validate_mcq_options
from utils_episode import load_jsonl
from cli.validate_episode_jsonl import validate_episode_file


OPTION_LINE_RE = re.compile(r"^([A-Z])\)\s*(.+?)\s*$")


def test_cli_build_and_convert_emit_expected_files() -> None:
    temp_path = Path("tmp_test_runs") / "cli_build_convert"
    temp_path.mkdir(parents=True, exist_ok=True)
    episodes_path = temp_path / "episodes.jsonl"
    converted_dir = temp_path / "converted"

    build_cmd = [
        sys.executable,
        "cli/build_grpo_episodes.py",
        "--seeds",
        "seeds_llm_clean_C.jsonl",
        "--phrase_bank",
        "phrase_bank_tierC_all.json",
        "--out",
        str(episodes_path),
        "--variants_per_seed",
        "1",
        "--seed",
        "42",
    ]
    subprocess.run(build_cmd, check=True)
    assert episodes_path.exists()
    assert load_jsonl(episodes_path)
    source_summary = validate_episode_file(episodes_path)
    assert source_summary["mismatch_count"] == 0
    assert source_summary["unmappable_candidate_count"] == 0
    assert source_summary["duplicate_option_count"] == 0

    convert_cmd = [
        sys.executable,
        "cli/convert_episodes.py",
        "--input",
        str(episodes_path),
        "--out_dir",
        str(converted_dir),
    ]
    subprocess.run(convert_cmd, check=True)

    assert (converted_dir / "episode_sft.jsonl").exists()
    assert (converted_dir / "episode_dpo.jsonl").exists()
    assert (converted_dir / "episode_grpo_offline.jsonl").exists()
    assert (converted_dir / "episode_grpo_online.jsonl").exists()

    offline_rows = load_jsonl(converted_dir / "episode_grpo_offline.jsonl")
    assert offline_rows

    for row in offline_rows:
        options = []
        for line in row["prompt"].splitlines():
            match = OPTION_LINE_RE.match(line.strip())
            if match:
                options.append({"label": match.group(1), "text": match.group(2)})
        validate_mcq_options(options)
        for response in row["responses"]:
            selected_option = extract_main_selected_option(response["text"])
            final_answer = extract_main_final_answer(response["text"])
            assert selected_option is not None
            assert final_answer is not None
            assert selected_option == resolve_option_for_value(options, final_answer)
