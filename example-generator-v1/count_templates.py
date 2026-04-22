import argparse
import json
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count template frequencies in a JSONL file and write a report."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the JSONL file.",
    )
    parser.add_argument(
        "--field",
        default="template",
        help="JSON field name that holds the template string.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Directory to write the report. If omitted, prints to terminal only.",
    )
    parser.add_argument(
        "--out-name",
        default=None,
        help="Report filename when --out-dir is used. Defaults to template_counts_<input_stem>.txt",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Input not found: {input_path}")

    counts: Counter[str] = Counter()
    total_lines = 0
    missing_field = 0

    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total_lines += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            value = obj.get(args.field)
            if value is None:
                missing_field += 1
                continue
            counts[str(value)] += 1

    report_lines = [
        f"input: {input_path}",
        f"field: {args.field}",
        f"total_lines: {total_lines}",
        f"missing_field: {missing_field}",
        f"unique_templates: {len(counts)}",
        "",
        "counts:",
    ]
    for template, count in counts.most_common():
        report_lines.append(f"{count}\t{template}")

    report_text = "\n".join(report_lines)
    print(report_text)

    if args.out_dir:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_name = args.out_name or f"template_counts_{input_path.stem}.txt"
        out_path = out_dir / out_name
        out_path.write_text(report_text, encoding="utf-8")
        print(f"\nWrote report to {out_path}")


if __name__ == "__main__":
    main()
