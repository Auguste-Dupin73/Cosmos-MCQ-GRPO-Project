import argparse
import json
import re
from typing import Dict, List, Tuple

# ----------------------------
# Config: what to remove
# ----------------------------

# Multi-word phrases first (order matters)
PLACE_PHRASES = [
    "online siparişte",
    "online alışverişte",
    "online",
    "internetten",
    "bir sitede",
    "bir uygulamada",
    "mağazada",
    "markette",
    "pazarda",
    "kırtasiyede",
    "büfede",
    "kafede",
    "kantinde",
    "kütüphanede",
    "okulda",
    "sınıfta",
    "evde",
    "ev içinde",
    "dükkanda",
    "marketten",
    "pazardan",
]

# Build a single regex that matches these as whole words/phrases (case-insensitive).
# We avoid \b for Turkish edge cases by using whitespace/punctuation boundaries.
def compile_place_regex(phrases: List[str]) -> re.Pattern:
    escaped = sorted((re.escape(p) for p in phrases), key=len, reverse=True)
    # boundary: start or whitespace or punctuation
    left = r"(^|[\s\(\[\{,;:])"
    right = r"($|[\s\)\]\},;:\.!?\"])"
    # capture left boundary to preserve it, remove only the phrase
    pat = left + r"(" + "|".join(escaped) + r")" + right
    return re.compile(pat, flags=re.IGNORECASE)

PLACE_RE = compile_place_regex(PLACE_PHRASES)

PLACEHOLDER_RE = re.compile(r"\{[a-zA-Z_][a-zA-Z0-9_]*\}")

def normalize(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    s = s.replace(" ,", ",").replace(" .", ".").replace(" ?", "?").replace(" !", "!")
    return s

def placeholders(s: str) -> Tuple[str, ...]:
    return tuple(sorted(set(PLACEHOLDER_RE.findall(s))))

def remove_places(text: str) -> str:
    # Iteratively remove until no further match (handles repeated patterns)
    prev = None
    s = text
    while prev != s:
        prev = s
        s = PLACE_RE.sub(lambda m: m.group(1) + m.group(3), s)
    # Clean up doubled spaces, stray punctuation spacing
    return normalize(s)

def should_clean_key(key: str, include_keys: List[str], include_regexes: List[re.Pattern]) -> bool:
    if include_keys and key in include_keys:
        return True
    if include_regexes:
        return any(r.search(key) for r in include_regexes)
    return False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True, help="Input phrase bank JSON")
    ap.add_argument("--out", dest="out_path", required=True, help="Output cleaned phrase bank JSON")
    ap.add_argument("--dry_run", action="store_true", help="Print changes, still writes output")
    ap.add_argument(
        "--keys",
        nargs="*",
        default=[],
        help="Explicit keys to clean (optional). If empty, uses --key_regex."
    )
    ap.add_argument(
        "--key_regex",
        nargs="*",
        default=[r".*_story_start$", r".*_story_event$"],
        help="Regex patterns for keys to clean (default: *_story_start and *_story_event)"
    )
    args = ap.parse_args()

    with open(args.in_path, "r", encoding="utf-8") as f:
        bank: Dict[str, object] = json.load(f)

    include_regexes = [re.compile(p) for p in args.key_regex]

    changes = 0
    removed = 0

    for key, val in list(bank.items()):
        if not isinstance(val, list):
            continue
        if not should_clean_key(key, args.keys, include_regexes):
            continue
        if not val:
            continue

        new_list: List[str] = []
        seen = set()

        for s in val:
            if not isinstance(s, str):
                continue
            original = normalize(s)
            ph_before = placeholders(original)

            cleaned = remove_places(original)
            ph_after = placeholders(cleaned)

            # If placeholders changed, skip cleaning (safety)
            if ph_before != ph_after:
                cleaned = original

            cleaned = normalize(cleaned)

            # Drop if empty after cleaning
            if not cleaned:
                removed += 1
                continue

            if cleaned not in seen:
                new_list.append(cleaned)
                seen.add(cleaned)

            if cleaned != original:
                changes += 1
                if args.dry_run:
                    print(f"\nKEY: {key}")
                    print(f"  - {original}")
                    print(f"  + {cleaned}")

        bank[key] = new_list

    with open(args.out_path, "w", encoding="utf-8") as f:
        json.dump(bank, f, ensure_ascii=False, indent=2)

    print("\n[OK] Cleanup finished.")
    print(f"Changed lines: {changes}")
    print(f"Removed empty lines: {removed}")
    print(f"Wrote: {args.out_path}")

if __name__ == "__main__":
    main()
