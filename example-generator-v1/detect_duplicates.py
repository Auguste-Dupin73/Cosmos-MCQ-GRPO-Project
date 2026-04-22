import json
import hashlib
import argparse
from collections import defaultdict


def _freeze(obj, *, ignore_list_order: bool):
    """
    Convert obj into a canonical, hashable structure.
    - dict: keys sorted
    - list: either kept order OR sorted order (by canonical JSON) if ignore_list_order
    """
    if isinstance(obj, dict):
        return tuple((k, _freeze(obj[k], ignore_list_order=ignore_list_order)) for k in sorted(obj.keys()))
    if isinstance(obj, list):
        frozen_items = [_freeze(x, ignore_list_order=ignore_list_order) for x in obj]
        if ignore_list_order:
            # Sort by stable JSON representation of each item
            frozen_items.sort(key=lambda x: json.dumps(x, ensure_ascii=False, separators=(",", ":")))
        return tuple(frozen_items)
    return obj


def fingerprint(seed: dict, *, ignore_list_order: bool, template_scoped: bool) -> str:
    """
    Hash a seed deterministically.
    If template_scoped=True, duplicates are only considered within same template
    (effectively the same as including template in the hash key; but we do it explicitly).
    """
    tpl = seed.get("template", None)
    frozen = _freeze(seed, ignore_list_order=ignore_list_order)

    payload = {
        "template": tpl if template_scoped else "__ALL__",
        "seed": frozen,
    }
    blob = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def iter_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            yield i, json.loads(line), line


def write_jsonl(path: str, rows: list[dict]):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser(description="Detect and optionally remove duplicate seeds in a JSONL file.")
    ap.add_argument("--in_seeds", required=True, help="Input JSONL file")
    ap.add_argument("--ignore_list_order", action="store_true",
                    help="Treat lists as unordered (e.g., [2,3] == [3,2])")
    ap.add_argument("--template_scoped", action="store_true",
                    help="Only consider duplicates within the same template")
    ap.add_argument("--quarantine", help="Write duplicate seeds to this JSONL file (with metadata)")
    ap.add_argument("--dedupe", action="store_true",
                    help="Write a de-duplicated output file (keep-first occurrence)")
    ap.add_argument("--out_seeds", help="Output JSONL (required with --dedupe)")
    ap.add_argument("--keep", choices=["first", "last"], default="first",
                    help="Which duplicate to keep when --dedupe is used (default: first)")

    args = ap.parse_args()

    if args.dedupe and not args.out_seeds:
        raise SystemExit("--dedupe requires --out_seeds")

    # Track:
    # fp -> first occurrence line index (or last occurrence line index if keep=last)
    seen_line = {}
    seen_seed = {}  # fp -> seed dict
    dup_entries = []  # duplicates (for quarantine)
    dup_count = 0

    # For per-template stats
    dup_by_template = defaultdict(int)
    total_by_template = defaultdict(int)
    unique_by_template = defaultdict(int)

    # If keep=last, we need all rows; if keep=first, we can stream-build output.
    all_rows = []  # list of (line_idx, seed_dict)
    for line_idx, seed, _raw in iter_jsonl(args.in_seeds):
        tpl = seed.get("template", "<?>")
        total_by_template[tpl] += 1

        fp = fingerprint(
            seed,
            ignore_list_order=args.ignore_list_order,
            template_scoped=args.template_scoped
        )

        if fp in seen_line:
            dup_count += 1
            dup_by_template[tpl] += 1

            # Prepare quarantine record (include where it first appeared)
            q = dict(seed)
            q["_dup_of_line"] = seen_line[fp]
            q["_dup_line"] = line_idx
            q["_dup_template"] = tpl
            dup_entries.append(q)

            # keep=last: overwrite the stored occurrence
            if args.keep == "last":
                seen_line[fp] = line_idx
                seen_seed[fp] = seed
        else:
            seen_line[fp] = line_idx
            seen_seed[fp] = seed

        if args.dedupe and args.keep == "last":
            # need full set for stable reconstruction
            all_rows.append((line_idx, seed))
        elif args.dedupe and args.keep == "first":
            # for keep-first, we can stream-build later from seen_seed
            # but we still want template totals etc; no need to store all_rows.
            pass

    # Compute unique_by_template by re-walking kept set
    # (We can approximate: unique = total - dups per template, but only if duplicates are template-scoped.
    # Better: compute exactly from kept seeds.)
    kept_seeds = list(seen_seed.values())
    for s in kept_seeds:
        tpl = s.get("template", "<?>")
        unique_by_template[tpl] += 1

    # Print duplicate instances
    # (If you want every line pairing printed, uncomment below; currently prints a concise report.)
    if dup_count > 0:
        print(f"Found {dup_count} duplicate occurrences.")
        # Show up to first 20 quarantine entries as examples
        for ex in dup_entries[:20]:
            print(f"[DUP] line {ex['_dup_line']} duplicates line {ex['_dup_of_line']} | template={ex.get('template')}")
        if len(dup_entries) > 20:
            print(f"... and {len(dup_entries) - 20} more.")

    else:
        print("No duplicates found.")

    # Quarantine write
    if args.quarantine:
        if dup_entries:
            write_jsonl(args.quarantine, dup_entries)
            print(f"WROTE QUARANTINE: {args.quarantine} ({len(dup_entries)} rows)")
        else:
            # still write empty file? you decide; default: write an empty file for predictability
            write_jsonl(args.quarantine, [])
            print(f"WROTE QUARANTINE: {args.quarantine} (0 rows)")

    # Dedupe write
    if args.dedupe:
        if args.keep == "first":
            # keep-first: preserve order of first appearance
            # sort seen items by the line they first appeared
            kept = sorted(seen_seed.items(), key=lambda kv: seen_line[kv[0]])
            out_rows = [seed for _fp, seed in kept]
        else:
            # keep-last: preserve order of last appearance
            # sort by last-seen line index stored in seen_line
            kept = sorted(seen_seed.items(), key=lambda kv: seen_line[kv[0]])
            out_rows = [seed for _fp, seed in kept]

        write_jsonl(args.out_seeds, out_rows)
        print(f"WROTE DEDUPED: {args.out_seeds} ({len(out_rows)} rows kept)")

    # Report
    print("\n=== DUPLICATE REPORT ===")
    print(f"Input file           : {args.in_seeds}")
    print(f"Total seeds          : {sum(total_by_template.values())}")
    print(f"Duplicate occurrences: {dup_count}")
    print(f"Unique seeds kept    : {len(seen_seed)}")
    print(f"Ignore list order    : {args.ignore_list_order}")
    print(f"Template scoped      : {args.template_scoped}")
    if args.dedupe:
        print(f"Dedupe mode          : ON (keep={args.keep})")
        print(f"Output file          : {args.out_seeds}")
    else:
        print("Dedupe mode          : OFF")
    if args.quarantine:
        print(f"Quarantine file      : {args.quarantine}")
    print("\nDuplicates per template (occurrences):")
    for tpl in sorted(total_by_template.keys()):
        print(f"  - {tpl}: total={total_by_template[tpl]} unique={unique_by_template[tpl]} dup_occurrences={dup_by_template[tpl]}")
    print("========================")


if __name__ == "__main__":
    main()
