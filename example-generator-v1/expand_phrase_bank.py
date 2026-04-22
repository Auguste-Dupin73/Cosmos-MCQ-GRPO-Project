import argparse
import json
import re
from typing import Dict, List, Tuple, Set


# -----------------------------
# Safety / validation helpers
# -----------------------------

PLACEHOLDER_RE = re.compile(r"\{[a-zA-Z_][a-zA-Z0-9_]*\}")
# Turkish suffix hell detector: placeholder immediately followed by letters (incl Turkish chars)
SUFFIX_HELL_RE = re.compile(r"(\{[a-zA-Z_][a-zA-Z0-9_]*\})([a-zA-ZçğıöşüÇĞİÖŞÜ]+)")

MULTISPACE_RE = re.compile(r"\s+")


def normalize(s: str) -> str:
    s = s.strip()
    s = MULTISPACE_RE.sub(" ", s)
    # clean spaced punctuation
    s = s.replace(" ,", ",").replace(" .", ".").replace(" ?", "?").replace(" !", "!")
    return s


def placeholders_in(s: str) -> Set[str]:
    return set(PLACEHOLDER_RE.findall(s))


def has_suffix_hell(s: str) -> bool:
    return SUFFIX_HELL_RE.search(s) is not None


def safe_add(
    out: List[str],
    seen: Set[str],
    candidate: str,
    expected_placeholders: Set[str],
) -> None:
    cand = normalize(candidate)
    if not cand:
        return
    if cand in seen:
        return
    # placeholders must match exactly (no missing, no extra)
    if placeholders_in(cand) != expected_placeholders:
        return
    # reject suffix attachment to placeholders: {item}i, {name}nin, etc.
    if has_suffix_hell(cand):
        return
    out.append(cand)
    seen.add(cand)


# -----------------------------
# Expansion rules
# -----------------------------

# Controlled synonym substitutions: exact phrase -> alternatives (suffix-safe)
SYNONYMS: Dict[str, List[str]] = {
    "olur": ["olur", "vardır"],
    "alınır": ["alınır", "satın alınır"],
    "uygulanır": ["uygulanır", "yapılır"],
    "toplam": ["toplam", "genel"],
    "ödeme": ["ödeme", "tutar"],
}

# For some keys we can safely vary sentence openers without suffix risk
PREFIXES = ["", "{name} için ", "Bu durumda "]
# For some keys we can safely vary closers
SUFFIXES = ["", " Sonuç bulunur.", " Hesap yapılır."]


def synonym_expand(base: str) -> List[str]:
    """
    Apply conservative, exact-token substitutions.
    We only substitute whole-word matches for keys in SYNONYMS.
    """
    variants = [base]
    for token, alts in SYNONYMS.items():
        new_variants = []
        # whole-word boundary (Turkish chars included in \w; we use explicit boundaries)
        pat = re.compile(rf"(?<!\w){re.escape(token)}(?!\w)")
        for v in variants:
            if pat.search(v):
                for alt in alts:
                    new_variants.append(pat.sub(alt, v))
            else:
                new_variants.append(v)
        # de-dupe at each step
        variants = list(dict.fromkeys(new_variants))
    return variants


def split_sentences(base: str) -> List[str]:
    """
    Turn comma-separated clauses into 2–3 sentences if possible.
    This is very effective for story_event keys and stays suffix-safe.
    """
    parts = [p.strip() for p in base.split(",") if p.strip()]
    if len(parts) < 2:
        return [base]

    # Join first two as separate sentences; keep remaining as last sentence
    s1 = parts[0] + "."
    s2 = parts[1] + "."
    rest = " ".join(p + "." for p in parts[2:]) if len(parts) > 2 else ""
    if rest:
        return [f"{s1} {s2} {rest}".strip()]
    return [f"{s1} {s2}".strip()]


def reorder_clauses_three(base: str, order: Tuple[int, int, int]) -> str:
    """
    Reorder 3 clauses split by commas.
    If not exactly 3 clauses, returns base unchanged.
    """
    parts = [p.strip() for p in base.split(",") if p.strip()]
    if len(parts) != 3:
        return base
    a, b, c = parts
    arr = [a, b, c]
    return ", ".join([arr[order[0]], arr[order[1]], arr[order[2]]])


def generate_variants(base: str, mode: str) -> List[str]:
    """
    mode controls rule set intensity:
      - "light": synonyms + optional prefix/suffix
      - "event": light + split + clause reorder (3-clause only)
      - "reason": light + add short sentence
    """
    variants: List[str] = []

    # 1) synonyms
    syns = synonym_expand(base)

    # 2) optional prefix/suffix
    for s in syns:
        for pre in PREFIXES:
            for suf in SUFFIXES:
                variants.append(f"{pre}{s}{suf}".strip())

    # 3) event-specific transformations
    if mode == "event":
        extra = []
        # split commas into sentences
        for v in list(variants):
            extra.extend(split_sentences(v))
        variants.extend(extra)

        # reorder if exactly 3 clauses (common in event lines we designed)
        reorder_orders = [(0, 2, 1), (1, 0, 2), (2, 0, 1)]
        for v in list(variants):
            for ord3 in reorder_orders:
                variants.append(reorder_clauses_three(v, ord3))

    # 4) reason-specific: append a safe “bridge sentence”
    if mode == "reason":
        variants.extend([base + " Adımlar takip edilir.", base + " İşlem sırası kullanılır."])

    # Normalize + de-dupe (order preserving)
    variants = [normalize(v) for v in variants]
    variants = list(dict.fromkeys(variants))
    return variants


# -----------------------------
# Key selection
# -----------------------------

def infer_mode(key: str) -> str:
    # story_event tends to be the best target for splitting/reordering
    if key.endswith("_story_event") or key in ("vat_story_event", "discunit_story_event", "bundisc_story_event"):
        return "event"
    # reasoning lines can take small extra sentences
    if "_reason_" in key:
        return "reason"
    return "light"


# -----------------------------
# Main expansion
# -----------------------------

def expand_bank(bank: Dict[str, List[str]], keys: List[str], target_per_key: int) -> Tuple[Dict[str, List[str]], Dict[str, int]]:
    stats = {}
    out_bank = dict(bank)

    for key in keys:
        if key not in out_bank or not isinstance(out_bank[key], list) or not out_bank[key]:
            print(f"[WARN] Key missing/empty: {key}")
            continue

        originals = [normalize(s) for s in out_bank[key] if isinstance(s, str)]
        originals = list(dict.fromkeys([s for s in originals if s]))
        expected_ph = placeholders_in(originals[0])

        mode = infer_mode(key)

        expanded: List[str] = []
        seen: Set[str] = set()

        # keep originals first
        for s in originals:
            safe_add(expanded, seen, s, expected_ph)

        # expand until target
        # round-robin generate from each original to keep diversity
        idx = 0
        while len(expanded) < target_per_key and idx < 100000:
            base = originals[idx % len(originals)]
            for v in generate_variants(base, mode):
                safe_add(expanded, seen, v, expected_ph)
                if len(expanded) >= target_per_key:
                    break
            idx += 1

            # stop if we've exhausted meaningful growth
            if idx > len(originals) * 50 and len(expanded) < min(target_per_key, len(originals) + 10):
                break

        out_bank[key] = expanded
        stats[key] = len(expanded)

    return out_bank, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True, help="Input phrase bank JSON")
    ap.add_argument("--out", dest="out_path", required=True, help="Output phrase bank JSON")
    ap.add_argument("--keys", nargs="*", default=[], help="Keys to expand (default: auto Tier-B focus)")
    ap.add_argument("--target_per_key", type=int, default=40, help="Target number of phrases per key")
    args = ap.parse_args()

    with open(args.in_path, "r", encoding="utf-8") as f:
        bank = json.load(f)

    # Default: expand only the newest / high-ROI Tier-B keys (you can add more later)
    if not args.keys:
        args.keys = [
            # VAT
            "vat_story_start", "vat_story_event", "vat_question",
            "vat_reason_intro", "vat_reason_calc", "vat_reason_final",
            # Discounted unit total
            "discunit_story_start", "discunit_story_event", "discunit_question",
            "discunit_reason_intro", "discunit_reason_calc", "discunit_reason_final",
            # Bundle discount
            "bundisc_story_start", "bundisc_story_event", "bundisc_question",
            "bundisc_reason_intro", "bundisc_reason_calc", "bundisc_reason_final",
        ]

    new_bank, stats = expand_bank(bank, args.keys, args.target_per_key)

    with open(args.out_path, "w", encoding="utf-8") as f:
        json.dump(new_bank, f, ensure_ascii=False, indent=2)

    print("\n[OK] Expansion complete.")
    for k in args.keys:
        if k in stats:
            print(f"  {k}: {stats[k]} phrases")


if __name__ == "__main__":
    main()
