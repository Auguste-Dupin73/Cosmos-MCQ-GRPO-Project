#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import itertools
import json
import math
from typing import Any, Dict, List, Tuple

from schemas_seed import seed_from_dict, as_list_int  # existing in your repo
from validate_seeds import validate_seed              # existing in your repo


# ----------------------------
# IO
# ----------------------------

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ----------------------------
# Helpers
# ----------------------------

def _as_int_list(v: Any) -> List[int]:
    return as_list_int(v)

def _cart_prod(fields: Dict[str, List[int]]) -> List[Dict[str, int]]:
    keys = list(fields.keys())
    values = [fields[k] for k in keys]
    combos: List[Dict[str, int]] = []
    for tup in itertools.product(*values):
        combos.append({k: int(tup[i]) for i, k in enumerate(keys)})
    return combos

def _lcm(a: int, b: int) -> int:
    return abs(a*b) // math.gcd(a, b) if a and b else 0

def _lcm_many(xs: List[int]) -> int:
    cur = 1
    for x in xs:
        cur = _lcm(cur, x)
    return cur

def _minmax(xs: List[int]) -> Tuple[int, int]:
    return (min(xs), max(xs))


# ----------------------------
# Template-specific checks
# Each check returns:
#   (ok, failures)
# failures: list of (rule_name, combo_dict, details)
# ----------------------------

Failure = Tuple[str, Dict[str, int], str]


# ========== Tier A + Phase2 (A-ish) ==========

def check_produce_consume_sell(s: Dict[str, Any]) -> Tuple[bool, List[Failure]]:
    fields = {
        "produce": _as_int_list(s.get("produce", 0)),
        "use1": _as_int_list(s.get("use1", 0)),
        "use2": _as_int_list(s.get("use2", 0)),
        "price": _as_int_list(s.get("price", 0)),
    }
    fails: List[Failure] = []
    for c in _cart_prod(fields):
        if c["produce"] <= 0:
            fails.append(("produce_positive", c, "produce must be > 0"))
        if c["use1"] < 0 or c["use2"] < 0:
            fails.append(("use_nonnegative", c, "use1/use2 cannot be negative"))
        if c["price"] <= 0:
            fails.append(("price_positive", c, "price must be > 0"))
        if c["produce"] < (c["use1"] + c["use2"]):
            fails.append(("no_negative_leftover", c, "produce must be >= use1+use2"))
    return (len(fails) == 0, fails)

def check_remainder_after_loss(s: Dict[str, Any]) -> Tuple[bool, List[Failure]]:
    fields = {
        "start": _as_int_list(s.get("start", 0)),
        "lost": _as_int_list(s.get("lost", 0)),
    }
    fails: List[Failure] = []
    for c in _cart_prod(fields):
        if c["start"] <= 0:
            fails.append(("start_positive", c, "start must be > 0"))
        if c["lost"] < 0:
            fails.append(("lost_nonnegative", c, "lost cannot be negative"))
        if c["start"] <= c["lost"]:
            fails.append(("start_gt_lost", c, "start must be > lost"))
    return (len(fails) == 0, fails)

def check_equal_sharing(s: Dict[str, Any]) -> Tuple[bool, List[Failure]]:
    fields = {
        "total": _as_int_list(s.get("total", 0)),
        "people": _as_int_list(s.get("people", 1)),
    }
    fails: List[Failure] = []
    for c in _cart_prod(fields):
        if c["total"] <= 0:
            fails.append(("total_positive", c, "total must be > 0"))
        if c["people"] <= 0:
            fails.append(("people_positive", c, "people must be > 0"))
        if c["people"] != 0 and c["total"] % c["people"] != 0:
            fails.append(("divisible", c, "total must be divisible by people"))
    return (len(fails) == 0, fails)

def check_multi_step_add_sub(s: Dict[str, Any]) -> Tuple[bool, List[Failure]]:
    fields = {
        "start": _as_int_list(s.get("start", 0)),
        "add": _as_int_list(s.get("add", 0)),
        "sub": _as_int_list(s.get("sub", 0)),
    }
    fails: List[Failure] = []
    for c in _cart_prod(fields):
        if c["start"] < 0 or c["add"] < 0:
            fails.append(("start_add_nonnegative", c, "start/add cannot be negative"))
        if c["sub"] < 0:
            fails.append(("sub_nonnegative", c, "sub cannot be negative"))
        if (c["start"] + c["add"]) < c["sub"]:
            fails.append(("no_negative_result", c, "start+add must be >= sub"))
    return (len(fails) == 0, fails)

def check_unit_price_quantity(s: Dict[str, Any]) -> Tuple[bool, List[Failure]]:
    fields = {
        "price": _as_int_list(s.get("price", 0)),
        "qty": _as_int_list(s.get("qty", 0)),
    }
    fails: List[Failure] = []
    for c in _cart_prod(fields):
        if c["price"] <= 0:
            fails.append(("price_positive", c, "price must be > 0"))
        if c["qty"] <= 0:
            fails.append(("qty_positive", c, "qty must be > 0"))
    return (len(fails) == 0, fails)

def check_rate_time(s: Dict[str, Any]) -> Tuple[bool, List[Failure]]:
    fields = {
        "rate": _as_int_list(s.get("rate", 0)),
        "time": _as_int_list(s.get("time", 0)),
    }
    fails: List[Failure] = []
    for c in _cart_prod(fields):
        if c["rate"] <= 0:
            fails.append(("rate_positive", c, "rate must be > 0"))
        if c["time"] <= 0:
            fails.append(("time_positive", c, "time must be > 0"))
    return (len(fails) == 0, fails)

def check_ratio_scaling(s: Dict[str, Any]) -> Tuple[bool, List[Failure]]:
    fields = {
        "ratio_a": _as_int_list(s.get("ratio_a", 1)),
        "ratio_b": _as_int_list(s.get("ratio_b", 1)),
        "total": _as_int_list(s.get("total", 0)),
    }
    fails: List[Failure] = []
    for c in _cart_prod(fields):
        if c["ratio_a"] <= 0 or c["ratio_b"] <= 0:
            fails.append(("ratio_positive", c, "ratio parts must be > 0"))
            continue
        if c["total"] <= 0:
            fails.append(("total_positive", c, "total must be > 0"))
            continue
        ssum = c["ratio_a"] + c["ratio_b"]
        if c["total"] % ssum != 0:
            fails.append(("divisible", c, "total must be divisible by ratio_a+ratio_b"))
    # also validate ask_side is a/b if present
    ask_side = s.get("ask_side", "a")
    if ask_side not in ("a", "b"):
        fails.append(("ask_side", {}, "ask_side must be 'a' or 'b'"))
    return (len(fails) == 0, fails)

def check_sum_and_difference(s: Dict[str, Any]) -> Tuple[bool, List[Failure]]:
    fields = {
        "total": _as_int_list(s.get("total", 0)),
        "diff": _as_int_list(s.get("diff", 0)),
    }
    fails: List[Failure] = []
    for c in _cart_prod(fields):
        if c["total"] <= 0:
            fails.append(("total_positive", c, "total must be > 0"))
            continue
        if c["diff"] < 0:
            fails.append(("diff_nonnegative", c, "diff cannot be negative"))
            continue
        if c["diff"] >= c["total"]:
            fails.append(("diff_lt_total", c, "diff must be < total"))
            continue
        if (c["total"] + c["diff"]) % 2 != 0:
            fails.append(("parity_even", c, "(total+diff) must be even"))
    return (len(fails) == 0, fails)

def check_compare_difference(s: Dict[str, Any]) -> Tuple[bool, List[Failure]]:
    fields = {
        "a": _as_int_list(s.get("a", 0)),
        "b": _as_int_list(s.get("b", 0)),
    }
    fails: List[Failure] = []
    for c in _cart_prod(fields):
        if c["a"] < 0 or c["b"] < 0:
            fails.append(("nonnegative", c, "a and b cannot be negative"))
            continue
        if c["a"] <= c["b"]:
            fails.append(("a_gt_b", c, "a must be > b to keep difference positive"))
    return (len(fails) == 0, fails)

def check_reverse_operation(s: Dict[str, Any]) -> Tuple[bool, List[Failure]]:
    fields = {
        "end": _as_int_list(s.get("end", 0)),
        "add": _as_int_list(s.get("add", 0)),
    }
    fails: List[Failure] = []
    for c in _cart_prod(fields):
        if c["end"] <= 0:
            fails.append(("end_positive", c, "end must be > 0"))
            continue
        if c["add"] < 0:
            fails.append(("add_nonnegative", c, "add cannot be negative"))
            continue
        if c["end"] <= c["add"]:
            fails.append(("end_gt_add", c, "end must be > add so start stays positive"))
    return (len(fails) == 0, fails)


# ========== Tier B (same as before) ==========

def check_buy_two_items_total_cost(s: Dict[str, Any]) -> Tuple[bool, List[Failure]]:
    fields = {
        "price1": _as_int_list(s.get("price1", 0)),
        "qty1": _as_int_list(s.get("qty1", 0)),
        "price2": _as_int_list(s.get("price2", 0)),
        "qty2": _as_int_list(s.get("qty2", 0)),
    }
    fails: List[Failure] = []
    for c in _cart_prod(fields):
        if c["price1"] <= 0 or c["price2"] <= 0:
            fails.append(("price_positive", c, "price must be > 0"))
        if c["qty1"] <= 0 or c["qty2"] <= 0:
            fails.append(("qty_positive", c, "qty must be > 0"))
    return (len(fails) == 0, fails)

def check_change_from_payment(s: Dict[str, Any]) -> Tuple[bool, List[Failure]]:
    fields = {
        "price1": _as_int_list(s.get("price1", 0)),
        "qty1": _as_int_list(s.get("qty1", 0)),
        "price2": _as_int_list(s.get("price2", 0)),
        "qty2": _as_int_list(s.get("qty2", 0)),
        "paid": _as_int_list(s.get("paid", 0)),
    }
    fails: List[Failure] = []
    for c in _cart_prod(fields):
        if min(c["price1"], c["price2"], c["qty1"], c["qty2"]) <= 0:
            fails.append(("positive_inputs", c, "prices/qty must be > 0"))
            continue
        total = c["price1"] * c["qty1"] + c["price2"] * c["qty2"]
        if c["paid"] < total:
            fails.append(("paid_ge_total", c, f"paid={c['paid']} < total={total}"))
    return (len(fails) == 0, fails)

def check_add_then_share(s: Dict[str, Any]) -> Tuple[bool, List[Failure]]:
    fields = {
        "start": _as_int_list(s.get("start", 0)),
        "add": _as_int_list(s.get("add", 0)),
        "people": _as_int_list(s.get("people", 1)),
    }
    fails: List[Failure] = []
    for c in _cart_prod(fields):
        if c["people"] <= 0:
            fails.append(("people_positive", c, "people must be > 0"))
            continue
        total = c["start"] + c["add"]
        if total % c["people"] != 0:
            fails.append(("divisible", c, f"{total} not divisible by {c['people']}"))
    return (len(fails) == 0, fails)

def check_percentage_final(base: int, pct: int, sign: str) -> bool:
    mult = (100 - pct) if sign == "-" else (100 + pct)
    return (base * mult) % 100 == 0

def check_percentage_discount_final_price(s: Dict[str, Any]) -> Tuple[bool, List[Failure]]:
    fields = {
        "original_price": _as_int_list(s.get("original_price", 0)),
        "discount_pct": _as_int_list(s.get("discount_pct", 0)),
    }
    fails: List[Failure] = []
    for c in _cart_prod(fields):
        if c["original_price"] <= 0:
            fails.append(("price_positive", c, "original_price must be > 0"))
            continue
        if not (1 <= c["discount_pct"] <= 99):
            fails.append(("pct_range", c, "discount_pct must be 1..99"))
            continue
        if not check_percentage_final(c["original_price"], c["discount_pct"], "-"):
            fails.append(("integer_safe", c, "discount produces fraction"))
    return (len(fails) == 0, fails)

def check_percentage_increase_final_price(s: Dict[str, Any]) -> Tuple[bool, List[Failure]]:
    fields = {
        "original_price": _as_int_list(s.get("original_price", 0)),
        "increase_pct": _as_int_list(s.get("increase_pct", 0)),
    }
    fails: List[Failure] = []
    for c in _cart_prod(fields):
        if c["original_price"] <= 0:
            fails.append(("price_positive", c, "original_price must be > 0"))
            continue
        if c["increase_pct"] <= 0:
            fails.append(("pct_positive", c, "increase_pct must be > 0"))
            continue
        if not check_percentage_final(c["original_price"], c["increase_pct"], "+"):
            fails.append(("integer_safe", c, "increase produces fraction"))
    return (len(fails) == 0, fails)

def check_discounted_unit_total_cost(s: Dict[str, Any]) -> Tuple[bool, List[Failure]]:
    fields = {
        "unit_price": _as_int_list(s.get("unit_price", 0)),
        "discount_pct": _as_int_list(s.get("discount_pct", 0)),
        "qty": _as_int_list(s.get("qty", 0)),
    }
    fails: List[Failure] = []
    for c in _cart_prod(fields):
        if c["unit_price"] <= 0 or c["qty"] <= 0:
            fails.append(("positive_inputs", c, "unit_price/qty must be > 0"))
            continue
        if not (1 <= c["discount_pct"] <= 99):
            fails.append(("pct_range", c, "discount_pct must be 1..99"))
            continue
        if (c["unit_price"] * (100 - c["discount_pct"])) % 100 != 0:
            fails.append(("integer_safe", c, "discounted unit produces fraction"))
    return (len(fails) == 0, fails)

def check_vat_total_cost(s: Dict[str, Any]) -> Tuple[bool, List[Failure]]:
    fields = {
        "unit_price": _as_int_list(s.get("unit_price", 0)),
        "vat_pct": _as_int_list(s.get("vat_pct", 0)),
        "qty": _as_int_list(s.get("qty", 0)),
    }
    fails: List[Failure] = []
    for c in _cart_prod(fields):
        if c["unit_price"] <= 0 or c["qty"] <= 0:
            fails.append(("positive_inputs", c, "unit_price/qty must be > 0"))
            continue
        if c["vat_pct"] <= 0:
            fails.append(("pct_positive", c, "vat_pct must be > 0"))
            continue
        if (c["unit_price"] * (100 + c["vat_pct"])) % 100 != 0:
            fails.append(("integer_safe", c, "VAT unit produces fraction"))
    return (len(fails) == 0, fails)

def check_bundle_discount_total_cost(s: Dict[str, Any]) -> Tuple[bool, List[Failure]]:
    fields = {
        "unit_price": _as_int_list(s.get("unit_price", 0)),
        "qty": _as_int_list(s.get("qty", 0)),
        "discount_pct": _as_int_list(s.get("discount_pct", 0)),
    }
    fails: List[Failure] = []
    for c in _cart_prod(fields):
        if c["unit_price"] <= 0 or c["qty"] <= 0:
            fails.append(("positive_inputs", c, "unit_price/qty must be > 0"))
            continue
        if not (1 <= c["discount_pct"] <= 99):
            fails.append(("pct_range", c, "discount_pct must be 1..99"))
            continue
        gross = c["unit_price"] * c["qty"]
        if (gross * (100 - c["discount_pct"])) % 100 != 0:
            fails.append(("integer_safe", c, "basket discount produces fraction"))
    return (len(fails) == 0, fails)


# ========== Tier C (same as before) ==========

def check_discount_then_vat_total_cost(s: Dict[str, Any]) -> Tuple[bool, List[Failure]]:
    fields = {
        "unit_price": _as_int_list(s.get("unit_price", 0)),
        "discount_pct": _as_int_list(s.get("discount_pct", 0)),
        "vat_pct": _as_int_list(s.get("vat_pct", 0)),
        "qty": _as_int_list(s.get("qty", 0)),
    }
    fails: List[Failure] = []
    for c in _cart_prod(fields):
        if min(c["unit_price"], c["qty"]) <= 0:
            fails.append(("positive_inputs", c, "unit_price/qty must be > 0"))
            continue
        if not (0 <= c["discount_pct"] < 100):
            fails.append(("pct_range", c, "discount_pct must be 0..99"))
            continue
        if not (0 <= c["vat_pct"] < 100):
            fails.append(("vat_range", c, "vat_pct must be 0..99"))
            continue
        if (c["unit_price"] * (100 - c["discount_pct"])) % 100 != 0:
            fails.append(("integer_safe_discount", c, "discounted unit produces fraction"))
            continue
        disc_unit = (c["unit_price"] * (100 - c["discount_pct"])) // 100
        if (disc_unit * (100 + c["vat_pct"])) % 100 != 0:
            fails.append(("integer_safe_vat", c, "VAT-after-discount produces fraction"))
    return (len(fails) == 0, fails)

def check_two_items_then_bundle_discount_total_cost(s: Dict[str, Any]) -> Tuple[bool, List[Failure]]:
    fields = {
        "unit_price1": _as_int_list(s.get("unit_price1", 0)),
        "qty1": _as_int_list(s.get("qty1", 0)),
        "unit_price2": _as_int_list(s.get("unit_price2", 0)),
        "qty2": _as_int_list(s.get("qty2", 0)),
        "discount_pct": _as_int_list(s.get("discount_pct", 0)),
    }
    fails: List[Failure] = []
    for c in _cart_prod(fields):
        if min(c["unit_price1"], c["qty1"], c["unit_price2"], c["qty2"]) <= 0:
            fails.append(("positive_inputs", c, "unit_price/qty must be > 0"))
            continue
        if not (0 <= c["discount_pct"] < 100):
            fails.append(("pct_range", c, "discount_pct must be 0..99"))
            continue
        subtotal = c["unit_price1"] * c["qty1"] + c["unit_price2"] * c["qty2"]
        if (subtotal * (100 - c["discount_pct"])) % 100 != 0:
            fails.append(("integer_safe", c, "basket discount produces fraction"))
    return (len(fails) == 0, fails)

def check_discounted_total_then_change(s: Dict[str, Any]) -> Tuple[bool, List[Failure]]:
    fields = {
        "unit_price": _as_int_list(s.get("unit_price", 0)),
        "qty": _as_int_list(s.get("qty", 0)),
        "discount_pct": _as_int_list(s.get("discount_pct", 0)),
        "paid": _as_int_list(s.get("paid", 0)),
    }
    fails: List[Failure] = []
    for c in _cart_prod(fields):
        if min(c["unit_price"], c["qty"]) <= 0:
            fails.append(("positive_inputs", c, "unit_price/qty must be > 0"))
            continue
        if not (0 <= c["discount_pct"] < 100):
            fails.append(("pct_range", c, "discount_pct must be 0..99"))
            continue
        if c["paid"] <= 0:
            fails.append(("paid_positive", c, "paid must be > 0"))
            continue
        gross = c["unit_price"] * c["qty"]
        if (gross * (100 - c["discount_pct"])) % 100 != 0:
            fails.append(("integer_safe", c, "discounted total produces fraction"))
            continue
        disc_total = (gross * (100 - c["discount_pct"])) // 100
        if c["paid"] < disc_total:
            fails.append(("paid_ge_discounted_total", c, "paid < discounted_total"))
    return (len(fails) == 0, fails)

def check_transform_then_share(s: Dict[str, Any]) -> Tuple[bool, List[Failure]]:
    fields = {
        "unit_price": _as_int_list(s.get("unit_price", 0)),
        "qty": _as_int_list(s.get("qty", 0)),
        "discount_pct": _as_int_list(s.get("discount_pct", 0)),
        "people": _as_int_list(s.get("people", 1)),
    }
    fails: List[Failure] = []
    for c in _cart_prod(fields):
        if min(c["unit_price"], c["qty"]) <= 0:
            fails.append(("positive_inputs", c, "unit_price/qty must be > 0"))
            continue
        if not (0 <= c["discount_pct"] < 100):
            fails.append(("pct_range", c, "discount_pct must be 0..99"))
            continue
        if c["people"] <= 0:
            fails.append(("people_positive", c, "people must be > 0"))
            continue
        gross = c["unit_price"] * c["qty"]
        if (gross * (100 - c["discount_pct"])) % 100 != 0:
            fails.append(("integer_safe", c, "discounted total produces fraction"))
            continue
        disc_total = (gross * (100 - c["discount_pct"])) // 100
        if disc_total % c["people"] != 0:
            fails.append(("divisible", c, "discounted_total not divisible by people"))
    return (len(fails) == 0, fails)


CHECKERS = {
    # Tier A templates
    "produce_consume_sell": check_produce_consume_sell,
    "remainder_after_loss": check_remainder_after_loss,
    "equal_sharing": check_equal_sharing,
    "multi_step_add_sub": check_multi_step_add_sub,
    "unit_price_quantity": check_unit_price_quantity,
    "rate_time": check_rate_time,

    # Phase 2 templates
    "ratio_scaling": check_ratio_scaling,
    "sum_and_difference": check_sum_and_difference,
    "compare_difference": check_compare_difference,
    "reverse_operation": check_reverse_operation,

    # Tier B templates
    "buy_two_items_total_cost": check_buy_two_items_total_cost,
    "change_from_payment": check_change_from_payment,
    "add_then_share": check_add_then_share,
    "percentage_discount_final_price": check_percentage_discount_final_price,
    "percentage_increase_final_price": check_percentage_increase_final_price,
    "discounted_unit_total_cost": check_discounted_unit_total_cost,
    "vat_total_cost": check_vat_total_cost,
    "bundle_discount_total_cost": check_bundle_discount_total_cost,

    # Tier C templates
    "discount_then_vat_total_cost": check_discount_then_vat_total_cost,
    "two_items_then_bundle_discount_total_cost": check_two_items_then_bundle_discount_total_cost,
    "discounted_total_then_change": check_discounted_total_then_change,
    "transform_then_share": check_transform_then_share,
}


# ----------------------------
# Auto-fix (conservative)
# ----------------------------

def fix_seed(seed: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """
    Conservative strategy:
    - Filter lists to satisfy invariants across all combinations
    - For paid fields: raise/filter to ensure paid >= required max
    - Avoid inventing new random numbers unless absolutely necessary
    """
    s = copy.deepcopy(seed)
    notes: List[str] = []
    tpl = s.get("template")

    # ---------- Tier A fixes ----------
    if tpl == "produce_consume_sell":
        produce = _as_int_list(s.get("produce", 0))
        use1 = _as_int_list(s.get("use1", 0))
        use2 = _as_int_list(s.get("use2", 0))
        price = _as_int_list(s.get("price", 0))

        # Filter negatives / zeros
        use1 = [u for u in use1 if u >= 0]
        use2 = [u for u in use2 if u >= 0]
        price = [p for p in price if p > 0]
        if use1 != s.get("use1"):  # may be scalar; keep simple
            pass

        max_use = (max(use1) if use1 else 0) + (max(use2) if use2 else 0)
        good_produce = [p for p in produce if p > 0 and p >= max_use]
        if good_produce and isinstance(s.get("produce"), list) and good_produce != s["produce"]:
            s["produce"] = good_produce
            notes.append("produce filtered to ensure produce >= max(use1)+max(use2) and >0")
        if isinstance(s.get("use1"), list) and use1 and use1 != s["use1"]:
            s["use1"] = use1
            notes.append("use1 filtered to remove negatives")
        if isinstance(s.get("use2"), list) and use2 and use2 != s["use2"]:
            s["use2"] = use2
            notes.append("use2 filtered to remove negatives")
        if isinstance(s.get("price"), list) and price and price != s["price"]:
            s["price"] = price
            notes.append("price filtered to keep >0")

    if tpl == "remainder_after_loss":
        start = _as_int_list(s.get("start", 0))
        lost = _as_int_list(s.get("lost", 0))
        start = [x for x in start if x > 0]
        lost = [x for x in lost if x >= 0]
        if start and lost:
            min_start = min(start)
            good_lost = [l for l in lost if l < min_start]
            if isinstance(s.get("lost"), list) and good_lost and good_lost != s["lost"]:
                s["lost"] = good_lost
                notes.append("lost filtered so max(lost) < min(start)")
            if isinstance(s.get("start"), list) and start != s["start"]:
                s["start"] = start
                notes.append("start filtered to keep >0")

    if tpl == "equal_sharing":
        total = _as_int_list(s.get("total", 0))
        people = _as_int_list(s.get("people", 1))
        total = [t for t in total if t > 0]
        people = [p for p in people if p > 0]
        if total and people:
            # easiest stable fix: filter people to divisors of ALL totals (global condition)
            good_people = []
            for p in people:
                if all(t % p == 0 for t in total):
                    good_people.append(p)
            if isinstance(s.get("people"), list) and good_people and good_people != s["people"]:
                s["people"] = good_people
                notes.append("people filtered to divisors of ALL totals")
            elif (not good_people) and isinstance(s.get("total"), list):
                # fallback: filter totals to those divisible by lcm(people)
                l = _lcm_many(people) if people else 1
                good_total = [t for t in total if l != 0 and t % l == 0]
                if good_total and good_total != s["total"]:
                    s["total"] = good_total
                    notes.append("total filtered to multiples of lcm(people)")

    if tpl == "multi_step_add_sub":
        start = _as_int_list(s.get("start", 0))
        add = _as_int_list(s.get("add", 0))
        sub = _as_int_list(s.get("sub", 0))
        start = [x for x in start if x >= 0]
        add = [x for x in add if x >= 0]
        sub = [x for x in sub if x >= 0]
        if start and add and sub:
            limit = min(start) + min(add)
            good_sub = [x for x in sub if x <= limit]
            if isinstance(s.get("sub"), list) and good_sub and good_sub != s["sub"]:
                s["sub"] = good_sub
                notes.append("sub filtered to ensure min(start)+min(add) >= max(sub)")

    if tpl == "unit_price_quantity":
        price = [p for p in _as_int_list(s.get("price", 0)) if p > 0]
        qty = [q for q in _as_int_list(s.get("qty", 0)) if q > 0]
        if isinstance(s.get("price"), list) and price and price != s["price"]:
            s["price"] = price
            notes.append("price filtered to keep >0")
        if isinstance(s.get("qty"), list) and qty and qty != s["qty"]:
            s["qty"] = qty
            notes.append("qty filtered to keep >0")

    if tpl == "rate_time":
        rate = [r for r in _as_int_list(s.get("rate", 0)) if r > 0]
        time = [t for t in _as_int_list(s.get("time", 0)) if t > 0]
        if isinstance(s.get("rate"), list) and rate and rate != s["rate"]:
            s["rate"] = rate
            notes.append("rate filtered to keep >0")
        if isinstance(s.get("time"), list) and time and time != s["time"]:
            s["time"] = time
            notes.append("time filtered to keep >0")

    if tpl == "ratio_scaling":
        ra = [x for x in _as_int_list(s.get("ratio_a", 1)) if x > 0]
        rb = [x for x in _as_int_list(s.get("ratio_b", 1)) if x > 0]
        totals = [t for t in _as_int_list(s.get("total", 0)) if t > 0]

        if ra and rb and totals:
            sums = sorted({a + b for a in ra for b in rb})
            if not sums:
                return s, notes

            # Primary: if totals are not universally valid, replace with [lcm, 2*lcm, 3*lcm]
            all_totals_ok = all(all(tt % sm == 0 for sm in sums) for tt in totals)
            if not all_totals_ok:
                l = _lcm_many(sums)
                if l > 0:
                    s["total"] = [l, 2 * l, 3 * l]
                    notes.append(
                        f"ratio_scaling: total replaced with [lcm, 2*lcm, 3*lcm] = [{l}, {2 * l}, {3 * l}] "
                        "as primary fix to satisfy divisibility for all ratio combinations"
                    )
                    return s, notes

    if tpl == "sum_and_difference":
        total = [t for t in _as_int_list(s.get("total", 0)) if t > 0]
        diff = [d for d in _as_int_list(s.get("diff", 0)) if d >= 0]
        if total and diff:
            # filter diff to be < min(total) and parity-compatible with all totals is hard;
            # safest: filter diff values that satisfy BOTH constraints for ALL totals
            min_total = min(total)
            good_diff = []
            for d in diff:
                if d >= min_total:
                    continue
                if all(((t + d) % 2 == 0) and (d < t) for t in total):
                    good_diff.append(d)
            if isinstance(s.get("diff"), list) and good_diff and good_diff != s["diff"]:
                s["diff"] = good_diff
                notes.append("diff filtered so diff<total and (total+diff) even for ALL totals")

    if tpl == "compare_difference":
        a = [x for x in _as_int_list(s.get("a", 0)) if x >= 0]
        b = [x for x in _as_int_list(s.get("b", 0)) if x >= 0]
        if a and b:
            max_b = max(b)
            good_a = [x for x in a if x > max_b]
            if isinstance(s.get("a"), list) and good_a and good_a != s["a"]:
                s["a"] = good_a
                notes.append("a filtered so min(a) > max(b) (positive differences)")

    if tpl == "reverse_operation":
        end = [x for x in _as_int_list(s.get("end", 0)) if x > 0]
        add = [x for x in _as_int_list(s.get("add", 0)) if x >= 0]
        if end and add:
            max_add = max(add)
            good_end = [e for e in end if e > max_add]
            if isinstance(s.get("end"), list) and good_end and good_end != s["end"]:
                s["end"] = good_end
                notes.append("end filtered so min(end) > max(add) (start stays positive)")
            # also filter add to be < min(end)
            min_end = min(good_end) if good_end else min(end)
            good_add = [a for a in add if a < min_end]
            if isinstance(s.get("add"), list) and good_add and good_add != s["add"]:
                s["add"] = good_add
                notes.append("add filtered so max(add) < min(end)")

    # ---------- Tier B/C fixes (same spirit as earlier) ----------
    if tpl == "change_from_payment":
        p1s = _as_int_list(s.get("price1", 0))
        q1s = _as_int_list(s.get("qty1", 0))
        p2s = _as_int_list(s.get("price2", 0))
        q2s = _as_int_list(s.get("qty2", 0))
        max_total = max(p1*q1 + p2*q2 for p1, q1, p2, q2 in itertools.product(p1s, q1s, p2s, q2s))
        paid_list = _as_int_list(s.get("paid", 0))
        kept = [p for p in paid_list if p >= max_total]
        if not kept:
            s["paid"] = [max_total]
            notes.append(f"paid adjusted to [{max_total}] to satisfy paid>=total for all combos")
        elif len(kept) != len(paid_list):
            s["paid"] = kept
            notes.append("paid list filtered to values >= max_total")

    if tpl in ("add_then_share",):
        starts = _as_int_list(s.get("start", 0))
        adds = _as_int_list(s.get("add", 0))
        people = _as_int_list(s.get("people", 1))
        good_people = []
        for p in people:
            if p <= 0:
                continue
            if all((st + ad) % p == 0 for st, ad in itertools.product(starts, adds)):
                good_people.append(p)
        if isinstance(s.get("people"), list) and good_people and good_people != s["people"]:
            s["people"] = good_people
            notes.append("people filtered to divide (start+add) for all combos")

    if tpl in ("percentage_discount_final_price", "percentage_increase_final_price"):
        base_key = "original_price"
        pct_key = "discount_pct" if tpl == "percentage_discount_final_price" else "increase_pct"
        sign = "-" if tpl == "percentage_discount_final_price" else "+"

        bases = [b for b in _as_int_list(s.get(base_key, 0)) if b > 0]
        pcts = _as_int_list(s.get(pct_key, 0))

        # 1) Primary: filter percentages that are integer-safe for ALL bases
        good = []
        for pct in pcts:
            if tpl == "percentage_discount_final_price" and not (1 <= pct <= 99):
                continue
            if tpl == "percentage_increase_final_price" and pct <= 0:
                continue
            if bases and all((b * (100 - pct) if sign == "-" else b * (100 + pct)) % 100 == 0 for b in bases):
                good.append(pct)

        if good:
            if s.get(pct_key) != good:
                s[pct_key] = good
                notes.append(f"{pct_key} filtered to keep only integer-safe percentage values")

        # 2) Fallback (DISCOUNT ONLY): generate a safe percentage set if nothing survived
        elif tpl == "percentage_discount_final_price":
            # Condition: base * (100 - pct) divisible by 100 for ALL bases
            # Let k = 100 - pct
            candidate_ks = [80, 70, 60, 50, 40, 30, 20, 10]  # conservative, common discounts
            safe_pcts = []

            for k in candidate_ks:
                pct = 100 - k
                if 1 <= pct <= 99 and all((b * k) % 100 == 0 for b in bases):
                    safe_pcts.append(pct)
                if len(safe_pcts) >= 3:
                    break

            if safe_pcts:
                s[pct_key] = safe_pcts
                notes.append(f"{pct_key} replaced with safe generated set {safe_pcts}")
            else:
                # 3) Fallback: adjust original_price to a "money-friendly" grid (multiples of 100)
                # This is necessary when bases are coprime with 100 (e.g., 399, 599), making integer percent impossible.
                rounded = []
                for b in bases:
                    # round to nearest 100; ties round up
                    r = int(round(b / 100.0)) * 100
                    if r <= 0:
                        r = 100
                    rounded.append(r)

                # remove duplicates while preserving order
                dedup = []
                for x in rounded:
                    if x not in dedup:
                        dedup.append(x)

                # If rounding makes current pcts valid, keep pcts and just replace prices.
                if all((b * (100 - pct)) % 100 == 0 for b in dedup for pct in pcts if 1 <= pct <= 99):
                    s[base_key] = dedup
                    notes.append(f"{base_key} replaced with rounded-to-100 values {dedup} to enable integer discounts")

    # ---------- Tier C fixes ----------

    if tpl == "discounted_unit_total_cost":
        ups = [u for u in _as_int_list(s.get("unit_price", 0)) if u > 0]
        pcts = [p for p in _as_int_list(s.get("discount_pct", 0)) if 1 <= p <= 99]

        if ups and pcts:
            # 1) Primary: filter discount_pct to those valid for ALL unit_price values
            good_pcts = [pct for pct in pcts if all((u * (100 - pct)) % 100 == 0 for u in ups)]

            if good_pcts:
                # Always write back as list (even if input was scalar) for consistency
                if s.get("discount_pct") != good_pcts:
                    s["discount_pct"] = good_pcts
                    notes.append("discounted_unit_total_cost: discount_pct filtered for integer-safe discounted unit")
            else:
                # 2) Secondary: filter unit_price to those valid for at least one pct
                good_ups = [u for u in ups if any((u * (100 - pct)) % 100 == 0 for pct in pcts)]
                if good_ups:
                    if s.get("unit_price") != good_ups:
                        s["unit_price"] = good_ups
                        notes.append(
                            "discounted_unit_total_cost: unit_price filtered to allow integer-safe discount for some discount_pct")

                    # After filtering unit_price, re-filter discount_pct again
                    ups2 = [u for u in _as_int_list(s.get("unit_price", 0)) if u > 0]
                    good_pcts2 = [pct for pct in pcts if all((u * (100 - pct)) % 100 == 0 for u in ups2)]
                    if good_pcts2 and s.get("discount_pct") != good_pcts2:
                        s["discount_pct"] = good_pcts2
                        notes.append("discounted_unit_total_cost: discount_pct re-filtered after unit_price fix")
                else:
                    # 3) Fallback: generate a small set of always-safe discount pcts based on unit_price divisibility
                    # Condition: (u*(100-p))%100==0  <=>  u*(100-p) divisible by 100.
                    # A simple universal-safe set is to pick p values where (100-p) is a factor of 100/gcd(u,100) across ups.
                    # Conservative approach: choose discounts that make (100-p) in {50, 25, 20, 10} when possible.
                    candidate_ks = [50, 25, 20, 10]  # k = 100 - pct
                    safe_pcts = []
                    for k in candidate_ks:
                        pct = 100 - k
                        if 1 <= pct <= 99 and all((u * k) % 100 == 0 for u in ups):
                            safe_pcts.append(pct)

                    if safe_pcts:
                        s["discount_pct"] = safe_pcts
                        notes.append(f"discounted_unit_total_cost: discount_pct replaced with safe set {safe_pcts}")

    if tpl == "vat_total_cost":
        ups = [u for u in _as_int_list(s.get("unit_price", 0)) if u > 0]
        vps = _as_int_list(s.get("vat_pct", 0))
        good = []
        for v in vps:
            if v <= 0:
                continue
            if ups and all((u * (100 + v)) % 100 == 0 for u in ups):
                good.append(v)
        if isinstance(s.get("vat_pct"), list) and good and good != s["vat_pct"]:
            s["vat_pct"] = good
            notes.append("vat_pct filtered for integer-safe VAT unit")

    if tpl == "bundle_discount_total_cost":
        ups = [u for u in _as_int_list(s.get("unit_price", 0)) if u > 0]
        qtys = [q for q in _as_int_list(s.get("qty", 0)) if q > 0]
        pcts = _as_int_list(s.get("discount_pct", 0))

        if ups and qtys and pcts:
            # Need integer final total for ALL combos:
            # gross = unit_price * qty
            # gross*(100-discount_pct) must be divisible by 100

            def all_gross(_ups, _qtys):
                return sorted({u * q for u, q in itertools.product(_ups, _qtys)})

            gross_vals = all_gross(ups, qtys)

            # 1) Primary: filter existing discount_pct values that work for ALL gross values
            good = []
            for pct in pcts:
                if not (1 <= pct <= 99):
                    continue
                k = 100 - pct
                if all((g * k) % 100 == 0 for g in gross_vals):
                    good.append(pct)

            if isinstance(s.get("discount_pct"), list) and good and good != s["discount_pct"]:
                s["discount_pct"] = good
                notes.append("discount_pct filtered for integer-safe basket discount")

            # 2) Fallback A: if nothing survived, generate a small safe set of discounts (more variety than amputate)
            if isinstance(s.get("discount_pct"), list) and not good:
                candidate_ks = [80, 70, 60, 50, 40, 30, 20, 10]  # k = 100 - pct
                safe_pcts = []
                for k in candidate_ks:
                    pct = 100 - k
                    if 1 <= pct <= 99 and all((g * k) % 100 == 0 for g in gross_vals):
                        safe_pcts.append(pct)
                    if len(safe_pcts) >= 3:
                        break

                if safe_pcts:
                    s["discount_pct"] = safe_pcts
                    notes.append(f"bundle_discount_total_cost: discount_pct replaced with safe generated set {safe_pcts}")
                else:
                    # 3) Fallback B: adjust unit_price to a "money-friendly" grid, then re-generate discounts
                    grids = [10, 20, 50, 100]  # smallest change first

                    def round_to_grid(x, g):
                        return max(g, int(round(x / float(g))) * g)

                    def dedup_preserve(seq):
                        out = []
                        for v in seq:
                            if v not in out:
                                out.append(v)
                        return out

                    fixed = False
                    for g in grids:
                        rups = dedup_preserve([round_to_grid(u, g) for u in ups])
                        rgross = all_gross(rups, qtys)

                        candidate_ks2 = [80, 70, 60, 50, 40, 30, 20, 10]
                        safe_pcts2 = []
                        for k in candidate_ks2:
                            pct = 100 - k
                            if 1 <= pct <= 99 and all((gg * k) % 100 == 0 for gg in rgross):
                                safe_pcts2.append(pct)
                            if len(safe_pcts2) >= 3:
                                break

                        if safe_pcts2:
                            s["unit_price"] = rups
                            s["discount_pct"] = safe_pcts2
                            notes.append(
                                f"bundle_discount_total_cost: unit_price rounded to grid {g} -> {rups}; "
                                f"discount_pct set to {safe_pcts2}"
                            )
                            fixed = True
                            break

                    if not fixed:
                        notes.append("bundle_discount_total_cost: could not generate integer-safe discounts even after unit_price rounding")

    if tpl == "discounted_total_then_change":
        ups = [u for u in _as_int_list(s.get("unit_price", 0)) if u > 0]
        qtys = [q for q in _as_int_list(s.get("qty", 0)) if q > 0]
        dps = _as_int_list(s.get("discount_pct", 0))
        paids = _as_int_list(s.get("paid", 0))

        good_dps = []
        for d in dps:
            if not (0 <= d < 100):
                continue
            ok_all = True
            for u, q in itertools.product(ups, qtys):
                gross = u*q
                if (gross * (100 - d)) % 100 != 0:
                    ok_all = False
                    break
            if ok_all:
                good_dps.append(d)
        if isinstance(s.get("discount_pct"), list) and good_dps and good_dps != s["discount_pct"]:
            s["discount_pct"] = good_dps
            notes.append("discount_pct filtered for integer-safe discounted total")

        # Raise/filter paid to cover max discounted total
        max_disc_total = 0
        for u, q, d in itertools.product(ups, qtys, _as_int_list(s.get("discount_pct", 0))):
            gross = u*q
            disc_total = (gross * (100 - d)) // 100
            max_disc_total = max(max_disc_total, disc_total)
        kept = [p for p in paids if p >= max_disc_total]
        if not kept:
            s["paid"] = [max_disc_total]
            notes.append(f"paid adjusted to [{max_disc_total}] to satisfy paid>=discounted_total")
        elif len(kept) != len(paids):
            s["paid"] = kept
            notes.append("paid filtered to values >= max_discounted_total")

    if tpl == "transform_then_share":
        ups = [u for u in _as_int_list(s.get("unit_price", 0)) if u > 0]
        qtys = [q for q in _as_int_list(s.get("qty", 0)) if q > 0]
        dps = _as_int_list(s.get("discount_pct", 0))
        people = [p for p in _as_int_list(s.get("people", 1)) if p > 0]

        good_dps = []
        for d in dps:
            if not (0 <= d < 100):
                continue
            ok_all = True
            for u, q in itertools.product(ups, qtys):
                gross = u*q
                if (gross * (100 - d)) % 100 != 0:
                    ok_all = False
                    break
            if ok_all:
                good_dps.append(d)
        if isinstance(s.get("discount_pct"), list) and good_dps and good_dps != s["discount_pct"]:
            s["discount_pct"] = good_dps
            notes.append("discount_pct filtered for integer-safe discounted total")

        # Now filter people to divide discounted_total for ALL combos
        dps2 = _as_int_list(s.get("discount_pct", 0))
        good_people = []
        for p in people:
            ok_all = True
            for u, q, d in itertools.product(ups, qtys, dps2):
                gross = u*q
                disc_total = (gross * (100 - d)) // 100
                if disc_total % p != 0:
                    ok_all = False
                    break
            if ok_all:
                good_people.append(p)
        if isinstance(s.get("people"), list) and good_people and good_people != s["people"]:
            s["people"] = good_people
            notes.append("people filtered for divisibility of discounted_total")

    if tpl == "discount_then_vat_total_cost":
        ups = [u for u in _as_int_list(s.get("unit_price", 0)) if u > 0]
        dps = [d for d in _as_int_list(s.get("discount_pct", 0)) if 0 <= d < 100]
        vps = [v for v in _as_int_list(s.get("vat_pct", 0)) if 0 <= v <= 100]

        if ups and dps and vps:
            # We require:
            # 1) discounted_unit is integer: unit_price*(100-d) % 100 == 0
            # 2) VAT unit after discount is integer: discounted_unit*(100+v) % 100 == 0
            #
            # discounted_unit = unit_price*(100-d)/100

            # ---------- 0) Ensure discounted unit is integer first (discount-stage) ----------
            def discounted_unit_int(u, d):
                return (u * (100 - d)) % 100 == 0

            # 0A) filter discount_pct
            good_dps = [d for d in dps if all(discounted_unit_int(u, d) for u in ups)]
            if isinstance(s.get("discount_pct"), list) and good_dps and good_dps != s["discount_pct"]:
                s["discount_pct"] = good_dps
                dps = good_dps
                notes.append("discount_then_vat_total_cost: discount_pct filtered for integer discounted unit")

            # 0B) generate safe discounts if none survive
            if isinstance(s.get("discount_pct"), list) and not good_dps:
                candidate_ks = [90, 80, 75, 70, 60, 50, 40, 30, 20, 10]  # k = 100 - d
                safe_dps = []
                for k in candidate_ks:
                    d = 100 - k
                    if 0 <= d < 100 and all((u * k) % 100 == 0 for u in ups):
                        safe_dps.append(d)
                    if len(safe_dps) >= 3:
                        break
                if safe_dps:
                    s["discount_pct"] = safe_dps
                    dps = safe_dps
                    notes.append(
                        f"discount_then_vat_total_cost: discount_pct replaced with safe generated set {safe_dps}")

            # 0C) if still impossible, round unit_price to a grid and regenerate discounts
            if isinstance(s.get("discount_pct"), list) and not any(
                    all(discounted_unit_int(u, d) for u in ups) for d in dps):
                if isinstance(s.get("unit_price"), list):
                    grids = [10, 20, 50, 100]

                    def round_to_grid(x, g):
                        return max(g, int(round(x / float(g))) * g)

                    def dedup_preserve(seq):
                        out = []
                        for v in seq:
                            if v not in out:
                                out.append(v)
                        return out

                    candidate_ks = [90, 80, 75, 70, 60, 50, 40, 30, 20, 10]

                    for g in grids:
                        rup = dedup_preserve([round_to_grid(u, g) for u in ups])
                        safe_dps2 = []
                        for k in candidate_ks:
                            d = 100 - k
                            if 0 <= d < 100 and all((u * k) % 100 == 0 for u in rup):
                                safe_dps2.append(d)
                            if len(safe_dps2) >= 3:
                                break
                        if safe_dps2:
                            s["unit_price"] = rup
                            ups = rup
                            s["discount_pct"] = safe_dps2
                            dps = safe_dps2
                            notes.append(
                                f"discount_then_vat_total_cost: unit_price rounded to grid {g} -> {rup}; discount_pct set to {safe_dps2}")
                            break

            def vat_ok(u, d, v):
                if not discounted_unit_int(u, d):
                    return False
                du = (u * (100 - d)) // 100
                return (du * (100 + v)) % 100 == 0

            # ---------- 1) Primary: filter vat_pct to those valid for ALL (unit_price, discount_pct) combos ----------
            good_vps = []
            for v in vps:
                ok_all = True
                for u, d in itertools.product(ups, dps):
                    if not vat_ok(u, d, v):
                        ok_all = False
                        break
                if ok_all:
                    good_vps.append(v)

            if isinstance(s.get("vat_pct"), list) and good_vps and good_vps != s["vat_pct"]:
                s["vat_pct"] = good_vps
                notes.append("discount_then_vat_total_cost: vat_pct filtered for integer-safe VAT after discount")
                return s, notes  # done

            # ---------- 2) Fallback A: generate a small safe VAT set if nothing survived ----------
            if isinstance(s.get("vat_pct"), list) and not good_vps:
                # Candidate VAT rates: include common TR rates (1,8,18,20) and also multiples of 5 for integer-friendliness
                candidate_v = [1, 5, 8, 10, 15, 18, 20]
                safe_v = []
                for v in candidate_v:
                    if all(vat_ok(u, d, v) for u, d in itertools.product(ups, dps)):
                        safe_v.append(v)
                    if len(safe_v) >= 3:
                        break

                if safe_v:
                    s["vat_pct"] = safe_v
                    notes.append(f"discount_then_vat_total_cost: vat_pct replaced with safe generated set {safe_v}")
                    return s, notes

            # ---------- 3) Fallback B: adjust unit_price to satisfy existing (or generated) VAT ----------
            # If vat_pct is a single value (common case), we can often rescue by rounding unit_price to a needed multiple.
            # Derivation:
            # Need du*(100+v) % 100 == 0  <=> du divisible by 100/gcd(100+v,100)
            # And du = u*(100-d)/100 integer.
            # Equivalent condition on u:
            # u*(100-d) divisible by 100*(100/g), where g=gcd(100+v,100)  => 10000/g
            #
            # We'll attempt to round unit_price up to the nearest multiple that satisfies this for the first (d,v).
            if isinstance(s.get("unit_price"), list):
                import math

                # choose representative d and v (first values)
                d0 = dps[0]
                v0 = vps[0]

                g = math.gcd(100 + v0, 100)
                need = 10000 // g  # u*(100-d) must be divisible by this

                k = 100 - d0
                step = need // math.gcd(k, need)  # u must be multiple of 'step'

                # Round each unit_price up to nearest multiple of step
                rup = []
                for u in ups:
                    r = ((u + step - 1) // step) * step
                    rup.append(r)

                # Dedup preserve order
                out = []
                for x in rup:
                    if x not in out:
                        out.append(x)

                # Verify that with rounded prices, ALL (u,d,v) combos pass VAT
                if out and all(vat_ok(u, d, v0) for u, d in itertools.product(out, dps)):
                    s["unit_price"] = out
                    notes.append(
                        f"discount_then_vat_total_cost: unit_price rounded up to multiples of {step} to satisfy vat_pct={v0} after discount")
                    # After price change, try to re-filter vat_pct again to keep original if possible
                    good_vps2 = []
                    for v in vps:
                        if all(vat_ok(u, d, v) for u, d in itertools.product(out, dps)):
                            good_vps2.append(v)
                    if good_vps2:
                        s["vat_pct"] = good_vps2
                        notes.append("discount_then_vat_total_cost: vat_pct re-filtered after unit_price fix")

    if tpl == "two_items_then_bundle_discount_total_cost":
        p1s = [p for p in _as_int_list(s.get("unit_price1", 0)) if p > 0]
        q1s = [q for q in _as_int_list(s.get("qty1", 0)) if q > 0]
        p2s = [p for p in _as_int_list(s.get("unit_price2", 0)) if p > 0]
        q2s = [q for q in _as_int_list(s.get("qty2", 0)) if q > 0]
        dps = _as_int_list(s.get("discount_pct", 0))

        if p1s and q1s and p2s and q2s and dps:
            # Helper: compute all subtotals (cart combinations)
            def all_subtotals(_p1s, _q1s, _p2s, _q2s):
                return sorted({p1 * q1 + p2 * q2 for p1, q1, p2, q2 in itertools.product(_p1s, _q1s, _p2s, _q2s)})

            subtotals = all_subtotals(p1s, q1s, p2s, q2s)

            # 1) Primary: filter existing discount_pct values that work for ALL subtotals
            good_dps = []
            for d in dps:
                if not (0 <= d < 100):
                    continue
                k = 100 - d
                if all((st * k) % 100 == 0 for st in subtotals):
                    good_dps.append(d)

            if isinstance(s.get("discount_pct"), list) and good_dps and good_dps != s["discount_pct"]:
                s["discount_pct"] = good_dps
                notes.append(
                    "two_items_then_bundle_discount_total_cost: discount_pct filtered for integer-safe basket discount")

            # 2) Fallback A: if nothing survived, generate a small safe set of discounts
            if isinstance(s.get("discount_pct"), list) and not good_dps:
                candidate_ks = [80, 70, 60, 50, 40, 30, 20, 10]  # k = 100 - discount
                safe_dps = []
                for k in candidate_ks:
                    d = 100 - k
                    if 0 <= d < 100 and all((st * k) % 100 == 0 for st in subtotals):
                        safe_dps.append(d)
                    if len(safe_dps) >= 3:
                        break

                if safe_dps:
                    s["discount_pct"] = safe_dps
                    notes.append(
                        f"two_items_then_bundle_discount_total_cost: discount_pct replaced with safe generated set {safe_dps}")
                else:
                    # 3) Fallback B: adjust prices to a "money-friendly" grid, then re-generate discounts
                    # Try progressively coarser rounding grids (smallest change first).
                    grids = [10, 20, 50, 100]

                    def round_to_grid(x, g):
                        # nearest multiple of g; ties go up
                        return max(g, int(round(x / float(g))) * g)

                    def dedup_preserve(seq):
                        out = []
                        for v in seq:
                            if v not in out:
                                out.append(v)
                        return out

                    fixed = False
                    for g in grids:
                        rp1s = dedup_preserve([round_to_grid(p, g) for p in p1s])
                        rp2s = dedup_preserve([round_to_grid(p, g) for p in p2s])
                        rst = all_subtotals(rp1s, q1s, rp2s, q2s)

                        safe_dps2 = []
                        for k in candidate_ks:
                            d = 100 - k
                            if 0 <= d < 100 and all((st * k) % 100 == 0 for st in rst):
                                safe_dps2.append(d)
                            if len(safe_dps2) >= 3:
                                break

                        if safe_dps2:
                            # apply rounded prices + new discounts
                            if isinstance(s.get("unit_price1"), list) and s.get("unit_price1") != rp1s:
                                s["unit_price1"] = rp1s
                            if isinstance(s.get("unit_price2"), list) and s.get("unit_price2") != rp2s:
                                s["unit_price2"] = rp2s
                            s["discount_pct"] = safe_dps2

                            notes.append(
                                f"two_items_then_bundle_discount_total_cost: prices rounded to grid {g} "
                                f"(unit_price1={rp1s}, unit_price2={rp2s}) then discount_pct set to {safe_dps2}"
                            )
                            fixed = True
                            break

                    # If still not fixed, leave as-is (amputate logic can handle later)
                    if not fixed:
                        notes.append(
                            "two_items_then_bundle_discount_total_cost: could not generate integer-safe discounts even after price rounding")

    return s, notes


# ----------------------------
# Main CLI
# ----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_seeds", required=True, help="input seeds.jsonl")
    ap.add_argument("--out_seeds", help="output seeds.jsonl (only for --fix)")
    ap.add_argument("--fix", action="store_true", help="attempt conservative auto-fixes")
    ap.add_argument("--in_place", action="store_true", help="overwrite --in_seeds file with fixed output")
    ap.add_argument("--amputate", action="store_true",
                    help="drop seeds that remain invalid after all fix attempts (only with --fix)")
    ap.add_argument("--quarantine", help="write amputated seeds to this JSONL file (only with --fix)")
    ap.add_argument("--max_failures", type=int, default=5, help="max failures printed per seed")
    args = ap.parse_args()

    rows = load_jsonl(args.in_seeds)

    fixed_rows: List[Dict[str, Any]] = []
    bad_count = 0
    fixed_count = 0
    amputated_count = 0
    quarantine_rows: List[Dict[str, Any]] = []
    remaining_bad = 0  # bad seeds that still remain in output (not fixed, not amputated)

    def amputate(row: Dict[str, Any], reason: str) -> None:
        nonlocal amputated_count
        amputated_count += 1
        if args.quarantine:
            q = dict(row)
            q["_amputated_reason"] = reason
            quarantine_rows.append(q)

    for i, d in enumerate(rows):
        tpl = d.get("template", "UNKNOWN")

        # 1) parse + base validate (your canonical validator)
        try:
            seed_obj = seed_from_dict(d)
            validate_seed(seed_obj)
        except Exception as e:
            print(f"[{i}] template={tpl} BASE VALIDATION FAIL: {e}\n")
            print(f"    SEED_JSON: {json.dumps(d, ensure_ascii=False)}\n")

            bad_count += 1
            if not args.fix:
                fixed_rows.append(d)
                continue

            d2, notes = fix_seed(d)
            try:
                seed_obj2 = seed_from_dict(d2)
                validate_seed(seed_obj2)
                print(f"[{i}] template={tpl} FIXED (base): {notes}")
                fixed_rows.append(d2)
                fixed_count += 1
            except Exception as e2:
                print(f"[{i}] template={tpl} UNFIXABLE (base): {e2}\n")
                print(f"    SEED_JSON: {json.dumps(d, ensure_ascii=False)}\n")
                if args.amputate:
                    amputate(d, f"base_unfixable: {e2}")
                else:
                    fixed_rows.append(d)
                    remaining_bad += 1
            continue

        # 2) template-specific lint (more actionable than validator errors)
        checker = CHECKERS.get(tpl)
        if checker is None:
            fixed_rows.append(d)
            continue

        ok, fails = checker(d)
        if ok:
            fixed_rows.append(d)
            continue

        bad_count += 1
        print(f"\n[{i}] template={tpl} FAIL ({len(fails)} failing combos). Showing up to {args.max_failures}:")
        for rule, combo, detail in fails[: args.max_failures]:
            print(f"  - rule={rule} combo={combo} detail={detail}")

        if not args.fix:
            fixed_rows.append(d)
            continue

        # 3) conservative fix attempt, then re-validate
        d2, notes = fix_seed(d)
        try:
            seed_obj2 = seed_from_dict(d2)
            validate_seed(seed_obj2)
            ok2, fails2 = checker(d2)
            if ok2:
                print(f"[{i}] FIXED: {notes}")
                fixed_rows.append(d2)
                fixed_count += 1
            else:
                print(f"[{i}] FIX ATTEMPT FAILED (still {len(fails2)} failing combos): {notes}")
                if args.amputate:
                    reason = fails2[0][0] if fails2 else "unknown"
                    amputate(d, f"checker_unfixable: {reason}")
                else:
                    fixed_rows.append(d)
                    remaining_bad += 1
        except Exception as e2:
            print(f"[{i}] FIX ATTEMPT FAILED (base validator): {notes} -> {e2}")
            if args.amputate:
                amputate(d, f"fix_failed_base_validator: {e2}")
            else:
                fixed_rows.append(d)
                remaining_bad += 1

    print(f"\nDONE. bad={bad_count} fixed={fixed_count} total={len(rows)}")
    # ---- Report section (summary) ----
    unchanged = len(rows) - fixed_count
    mode = "FIX" if args.fix else "LINT"

    print("\n=== SEED DOCTOR REPORT ===")
    print(f"Mode        : {mode}")
    print(f"Total seeds  : {len(rows)}")
    print(f"Bad seeds    : {bad_count}")
    print(f"Fixed seeds  : {fixed_count}")
    print(f"Amputated   : {amputated_count}")
    print(f"Unchanged    : {unchanged}")
    print(f"Output seeds : {len(fixed_rows)}")
    if args.fix:
        print(f"Remain bad   : {remaining_bad}")

    if bad_count == 0:
        print("Status      : ✅ No bad seeds found.")
    else:
        if args.fix:
            if remaining_bad == 0:
                print("Status      : ✅ All bad seeds were fixed/amputated.")
            else:
                print("Status      : ⚠️ Some seeds were bad; fix attempted.")
        else:
            print("Status      : ❌ Bad seeds found (no fixes applied; run with --fix).")

    if args.fix:
        if args.in_place:
            print(f"Output file  : {args.in_seeds}")
        elif args.out_seeds:
            print(f"Output file  : {args.out_seeds}")
    print("==========================\n")

    if bad_count > 0 and not args.fix:
        raise SystemExit(1)

    if args.fix:
        if args.in_place:
            if args.in_place and remaining_bad > 0:
                raise SystemExit("Refusing in-place overwrite: some bad seeds remain in output.")
            write_jsonl(args.in_seeds, fixed_rows)
            print(f"OVERWROTE: {args.in_seeds}")

            # Write amputated seeds, if requested (only meaningful in --fix mode)
            if args.quarantine and quarantine_rows:
                write_jsonl(args.quarantine, quarantine_rows)
                print(f"WROTE QUARANTINE: {args.quarantine}")
        else:
            if not args.out_seeds:
                raise SystemExit("--fix requires --out_seeds (or use --in_place)")
            write_jsonl(args.out_seeds, fixed_rows)
            print(f"WROTE: {args.out_seeds}")

            # Write amputated seeds, if requested (only meaningful in --fix mode)
            if args.quarantine and quarantine_rows:
                write_jsonl(args.quarantine, quarantine_rows)
                print(f"WROTE QUARANTINE: {args.quarantine}")


if __name__ == "__main__":
    main()
