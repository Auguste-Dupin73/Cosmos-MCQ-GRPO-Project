import argparse
import json
import itertools
from copy import deepcopy
from typing import Any, Dict, List, Tuple, Optional

from schemas_seed import seed_from_dict
from validate_seeds import validate_seed


# Keys that are not part of the combinational domain (singletons / meta)
NON_DOMAIN_KEYS = {"template", "currency"}

def is_list_value(v: Any) -> bool:
    return isinstance(v, list)

def domain_keys(d: Dict[str, Any]) -> List[str]:
    return [k for k, v in d.items() if k not in NON_DOMAIN_KEYS and is_list_value(v)]

def build_assignment_dict(seed_dict: Dict[str, Any], assignment: Dict[str, Any]) -> Dict[str, Any]:
    """Return a dict where list fields are replaced by the chosen singleton assignment."""
    out = deepcopy(seed_dict)
    for k, v in assignment.items():
        out[k] = v
    return out

def validate_assignment(seed_dict: Dict[str, Any], assignment: Dict[str, Any]) -> bool:
    """Validate one concrete assignment using existing validate_seed logic."""
    try:
        concrete = build_assignment_dict(seed_dict, assignment)
        s = seed_from_dict(concrete)
        validate_seed(s)
        return True
    except Exception:
        return False

def iter_other_products(domains: Dict[str, List[Any]], keys: List[str], fixed_key: str) -> itertools.product:
    other_keys = [k for k in keys if k != fixed_key]
    other_lists = [domains[k] for k in other_keys]
    return other_keys, itertools.product(*other_lists)

def has_support(seed_dict: Dict[str, Any], domains: Dict[str, List[Any]], keys: List[str], fixed_key: str, val: Any) -> bool:
    """Does fixed_key=val have at least one valid completion over other domains?"""
    other_keys, prod = iter_other_products(domains, keys, fixed_key)
    for combo in prod:
        assignment = {fixed_key: val}
        assignment.update({k: combo[i] for i, k in enumerate(other_keys)})
        if validate_assignment(seed_dict, assignment):
            return True
    return False

def arc_prune(seed_dict: Dict[str, Any], domains: Dict[str, List[Any]]) -> Tuple[Dict[str, List[Any]], bool]:
    """Iteratively remove values that have no support (arc consistency-ish)."""
    keys = list(domains.keys())
    changed_any = False

    changed = True
    while changed:
        changed = False
        for k in keys:
            kept = []
            for val in domains[k]:
                if has_support(seed_dict, domains, keys, k, val):
                    kept.append(val)
            if len(kept) != len(domains[k]):
                domains[k] = kept
                changed = True
                changed_any = True

            if not domains[k]:
                # Domain wiped out: impossible seed
                return domains, changed_any

    return domains, changed_any

def all_combos_valid(seed_dict: Dict[str, Any], domains: Dict[str, List[Any]]) -> bool:
    keys = list(domains.keys())
    for combo in itertools.product(*(domains[k] for k in keys)):
        assignment = {keys[i]: combo[i] for i in range(len(keys))}
        if not validate_assignment(seed_dict, assignment):
            return False
    return True

def find_first_invalid(seed_dict: Dict[str, Any], domains: Dict[str, List[Any]]) -> Optional[Dict[str, Any]]:
    keys = list(domains.keys())
    for combo in itertools.product(*(domains[k] for k in keys)):
        assignment = {keys[i]: combo[i] for i in range(len(keys))}
        if not validate_assignment(seed_dict, assignment):
            return assignment
    return None

def greedy_fix(seed_dict: Dict[str, Any], domains: Dict[str, List[Any]], max_steps: int = 200) -> bool:
    """
    If arc pruning leaves some invalid combos, remove values greedily until all combos are valid.
    Strategy:
      - find an invalid assignment
      - remove one value involved, preferring keys with larger domains (least destructive heuristic)
    """
    for _ in range(max_steps):
        bad = find_first_invalid(seed_dict, domains)
        if bad is None:
            return True

        # Choose a key to prune: largest domain first (least brittle)
        candidates = sorted(domains.keys(), key=lambda k: len(domains[k]), reverse=True)
        pruned = False
        for k in candidates:
            val = bad[k]
            if val in domains[k] and len(domains[k]) > 1:
                domains[k].remove(val)
                pruned = True
                break
        if not pruned:
            # Can't prune without wiping a domain; give up
            return False

        # After each removal, re-arc-prune to propagate constraints
        domains, _ = arc_prune(seed_dict, domains)

        # If any domain is empty, fail
        if any(len(v) == 0 for v in domains.values()):
            return False

    return False

def prune_seed(seed_dict: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Returns (pruned_seed_dict, report)
    """
    d = deepcopy(seed_dict)
    keys = domain_keys(d)
    domains = {k: list(d[k]) for k in keys}

    # 1) Arc prune
    domains, arc_changed = arc_prune(d, domains)

    report = {
        "template": d.get("template"),
        "arc_pruned": arc_changed,
        "domains_before": {k: list(seed_dict[k]) for k in keys},
        "domains_after_arc": {k: list(domains[k]) for k in keys},
        "greedy_used": False,
        "success": False,
    }

    if any(len(v) == 0 for v in domains.values()):
        report["success"] = False
        report["error"] = "A domain became empty during arc pruning. No valid combinations exist with current lists."
        return d, report

    # 2) If still not fully valid, greedy fix
    if not all_combos_valid(d, domains):
        ok = greedy_fix(d, domains)
        report["greedy_used"] = True
        if not ok:
            report["success"] = False
            report["error"] = "Could not fully eliminate invalid combinations without wiping a domain."
            report["domains_after_greedy"] = {k: list(domains[k]) for k in keys}
            return d, report

    report["success"] = True
    report["domains_final"] = {k: list(domains[k]) for k in keys}

    # Write back pruned lists
    for k in keys:
        d[k] = domains[k]

    return d, report

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True, help="Input JSONL seeds file")
    ap.add_argument("--out", dest="out_path", required=True, help="Output JSONL seeds file (pruned)")
    ap.add_argument("--report", dest="report_path", default="", help="Optional JSON report output")
    args = ap.parse_args()

    reports = []
    out_lines = []

    with open(args.in_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            seed_dict = json.loads(line)
            pruned, rep = prune_seed(seed_dict)
            reports.append(rep)
            out_lines.append(json.dumps(pruned, ensure_ascii=False))

    with open(args.out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines) + "\n")

    if args.report_path:
        with open(args.report_path, "w", encoding="utf-8") as f:
            json.dump(reports, f, ensure_ascii=False, indent=2)

    ok = sum(1 for r in reports if r.get("success"))
    print(f"[OK] Wrote pruned seeds to: {args.out_path}")
    print(f"Successful prunes: {ok}/{len(reports)}")
    if args.report_path:
        print(f"[OK] Wrote report to: {args.report_path}")

if __name__ == "__main__":
    main()
