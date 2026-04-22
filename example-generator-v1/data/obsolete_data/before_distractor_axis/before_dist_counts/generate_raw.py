from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from dataclasses import asdict
from typing import Any, Dict, List

from schemas_seed import (
    Seed,
    seed_from_dict,
    as_list_int,
    as_list_str,
    ProduceConsumeSellSeed,
    RemainderAfterLossSeed,
    EqualSharingSeed,
    MultiStepAddSubSeed,
    UnitPriceQuantitySeed,
    RateTimeSeed,
    RatioScalingSeed,
    SumAndDifferenceSeed,
    CompareDifferenceSeed,
    ReverseOperationSeed,
    BuyTwoItemsTotalCostSeed,
    ChangeFromPaymentSeed,
    AddThenShareSeed,
    PercentageDiscountFinalPriceSeed,
    PercentageIncreaseFinalPriceSeed,
    DiscountedUnitTotalCostSeed,
    VatTotalCostSeed,
    BundleDiscountTotalCostSeed,
    DiscountThenVatTotalCostSeed,
    TwoItemsThenBundleDiscountTotalCostSeed,
    DiscountedTotalThenChangeSeed,
    TransformThenShareSeed
)
from validate_seeds import validate_seed


def pick(rng: random.Random, lst: List[Any]) -> Any:
    return lst[rng.randrange(len(lst))]


def pick_int(rng: random.Random, x) -> int:
    xs = as_list_int(x)
    return int(pick(rng, xs))


def pick_str(rng: random.Random, x) -> str:
    xs = as_list_str(x)
    return str(pick(rng, xs))


def apply_intent_skin(text: str, rng: random.Random, pb: Dict[str, Any], enabled: bool, prob: float) -> tuple[str, bool]:
    """Prompt-only intent skins (kampanya/etiket/kasada/fiş). No schema/logic changes."""
    if not enabled:
        return (text, False)
    # Clamp prob
    if prob < 0.0:
        prob = 0.0
    if prob > 1.0:
        prob = 1.0
    if prob < 1.0 and rng.random() >= prob:
        return (text, False)

    lines = pb.get("intent_story_line") or []
    if not lines:
        return (text, False)

    marker = "\nCevap:"
    if marker not in text:
        return (f"{pick_str(rng, lines)} {text}", True)

    q, rest = text.split(marker, 1)
    intent = pick_str(rng, lines)
    q2 = f"{intent} {q.strip()}"
    return (q2 + marker + rest, True)






def apply_distractor_axis(text: str, rng: random.Random, pb: Dict[str, Any], template: str | None, enabled: bool, prob: float) -> tuple[str, bool]:
    """Prompt-only distractor axis: inject one extra, semantically-plausible sentence containing a number
    that is NOT used in the gold computation. Conservative: one sentence, applied with probability prob."""
    if not enabled:
        return (text, False)

    # Clamp prob
    if prob < 0.0:
        prob = 0.0
    if prob > 1.0:
        prob = 1.0
    if prob < 1.0 and rng.random() >= prob:
        return (text, False)

    key = f"distractor_story_line__{template}" if template else "distractor_story_line"
    lines = pb.get(key) or []
    if not lines and key != "distractor_story_line":
        lines = pb.get("distractor_story_line") or []
    if not lines:
        return (text, False)

    marker = "\nCevap:"
    if marker in text:
        q, rest = text.split(marker, 1)
    else:
        q, rest = text, ""

    # Try to avoid reusing numbers already present in the prompt (non-suffix-safe-ish).
    existing = set(int(x) for x in re.findall(r"\d+", q))
    distractor_num = None
    for _ in range(50):
        cand = rng.randrange(5, 501, 5)  # 5..500 step 5 (nice-looking prices/amounts)
        if cand not in existing:
            distractor_num = cand
            break
    if distractor_num is None:
        distractor_num = rng.randrange(5, 501, 5)

    line = pick_str(rng, lines)
    try:
        injected = line.format(distractor_num=distractor_num, currency="TL")
    except Exception:
        # If user edits the phrase bank with extra placeholders, fail closed.
        return (text, False)

    q2 = f"{q.strip()} {injected}".strip()
    if rest:
        return (q2 + marker + rest, True)
    return (q2, True)

# -----------------------------
# Tier C semantic completeness guardrails
# -----------------------------

_PLACEHOLDER_RE = re.compile(r"{([a-zA-Z_][a-zA-Z0-9_]*)}")

def _placeholders_in(fmt: str) -> set[str]:
    return set(_PLACEHOLDER_RE.findall(fmt or ""))

def pick_fmt_with_required(rng: random.Random, fmts: List[str], required_vars: set[str]) -> str | None:
    """Pick a format string that explicitly mentions all required placeholders."""
    if not fmts:
        return None
    # Filter candidates that contain all required placeholders
    candidates = [s for s in fmts if required_vars.issubset(_placeholders_in(s))]
    if not candidates:
        return None
    return pick(rng, candidates)

# Required stated variables per Tier C template + ask-target.
# These are the *numeric premises* that must appear in the story_event to avoid underdetermined prompts.
_TIERC_REQUIRED_EVENT_VARS: Dict[str, Dict[str, set[str]]] = {
    "discount_then_vat_total_cost": {
        # asking for total: we must state enough to compute it
        "total": {"unit_price", "discount_pct", "vat_pct", "qty"},
        # asking for qty: we must state total and transformation rates
        "qty": {"unit_price", "discount_pct", "vat_pct", "total"},
    },
    "two_items_then_bundle_discount_total_cost": {
        "final_total": {"unit_price1", "qty1", "unit_price2", "qty2", "discount_pct"},
        "discount_pct": {"subtotal", "final_total"},
    },
    "discounted_total_then_change": {
        "change": {"unit_price", "qty", "discount_pct", "paid"},
        "paid": {"discounted_total", "change"},
    },
    "transform_then_share": {
        "each": {"unit_price", "qty", "discount_pct", "people"},
        "people": {"discounted_total", "each"},
    },
}

def tierc_required_event_vars(template: str, ask_target: str) -> set[str]:
    return _TIERC_REQUIRED_EVENT_VARS.get(template, {}).get(ask_target, set())

# -----------------------------
# Unknown-position variants (Tier C only)
# -----------------------------

_TIERC_ASK_TARGETS = {
    # default: "total"
    "discount_then_vat_total_cost": ["total", "qty"],
    # default: "final_total"
    "two_items_then_bundle_discount_total_cost": ["final_total", "discount_pct"],
    # default: "change"
    "discounted_total_then_change": ["change", "paid"],
    # default: "each"
    "transform_then_share": ["each", "people"],
}

def choose_tierc_ask_target(template: str, rng: random.Random, enabled: bool, unknown_prob: float) -> str:
    """Tier C unknown-position axis controller.

    - If disabled: always return the default ask-target for the template.
    - If enabled: return default with probability (1 - unknown_prob), otherwise sample from non-default targets.
    """
    # Clamp for safety
    if unknown_prob < 0.0:
        unknown_prob = 0.0
    if unknown_prob > 1.0:
        unknown_prob = 1.0

    # Default targets (must match template semantics)
    if template == "discount_then_vat_total_cost":
        default = "total"
    elif template == "two_items_then_bundle_discount_total_cost":
        default = "final_total"
    elif template == "discounted_total_then_change":
        default = "change"
    elif template == "transform_then_share":
        default = "each"
    else:
        default = "default"

    if not enabled:
        return default

    opts = _TIERC_ASK_TARGETS.get(template) or [default]
    # Keep default unless we explicitly branch into unknown variant
    if rng.random() >= unknown_prob:
        return default

    alts = [x for x in opts if x != default]
    if not alts:
        return default
    return rng.choice(alts)


def seed_to_ctx(seed: Seed, rng: random.Random) -> Dict[str, Any]:
    d = asdict(seed)

    name = d.get("name", None)
    if name is None:
        d["name"] = "Ali"
    else:
        d["name"] = pick_str(rng, name)

    return d

def apply_skin(text: str, pb: Dict[str, Any], template: str, rng: random.Random) -> str:
    """
    Wraps the question text with a semantic skin prefix/suffix.
    Safe by construction: skins are independent strings (no placeholder suffixing).
    """
    template_skins = pb.get("template_skins", {})
    skins = pb.get("skins", {})

    options = template_skins.get(template, [])
    if not options:
        return text

    skin_name = rng.choice(options)
    skin = skins.get(skin_name, {})

    prefix_list = skin.get("prefix", [""])
    suffix_list = skin.get("suffix", [""])

    prefix = rng.choice(prefix_list) if prefix_list else ""
    suffix = rng.choice(suffix_list) if suffix_list else ""

    return f"{prefix}{text}{suffix}"


# -----------------------------
# Place / channel collision guardrail
# -----------------------------

# Lowercase match against text.lower()
_PLACE_WORDS = [
    "mağazada", "markette", "pazarda", "kırtasiyede", "büfede", "kafede", "kantinde",
    "kütüphanede", "okulda", "sınıfta", "evde",
    "marketten", "pazardan",
    "internetten", "online", "uygulamada", "sitede", "dükkanda"
]

def _count_place_cues(text: str) -> int:
    t = (text or "").lower()
    return sum(1 for w in _PLACE_WORDS if w in t)

def contains_place(text: str) -> bool:
    return _count_place_cues(text) >= 1

def has_place_collision(text: str) -> bool:
    return _count_place_cues(text) > 1

def build_question_with_optional_skin(parts: List[str], pb: Dict[str, Any], template: str, rng: random.Random) -> str:
    """Apply semantic skin ONLY to the first segment, and only if it is place-neutral."""
    if not parts:
        return ""
    start = parts[0]
    if not contains_place(start):
        start = apply_skin(start, pb, template, rng)
    return " ".join([start] + parts[1:])



def render_produce_consume_sell(seed: ProduceConsumeSellSeed, pb: Dict[str, Any], rng: random.Random) -> str:
    ctx = seed_to_ctx(seed, rng)

    ctx["produce"] = pick_int(rng, seed.produce)
    ctx["use1"] = pick_int(rng, seed.use1)
    ctx["use2"] = pick_int(rng, seed.use2)
    ctx["price"] = pick_int(rng, seed.price)

    purposes = pb.get("default_purposes", [["kahvaltıda", "börek"]])
    breakfast_ctx, dish = pick(rng, purposes)
    ctx["breakfast_ctx"] = ctx.get("breakfast_ctx") or breakfast_ctx
    ctx["dish"] = ctx.get("dish") or dish

    remain = ctx["produce"] - ctx["use1"] - ctx["use2"]
    total_income = remain * ctx["price"]
    ctx["remain"] = remain
    ctx["total"] = total_income

    parts = [
        pick(rng, pb["produce"]).format(**ctx),
        pick(rng, pb["consume_a"]).format(**ctx),
        pick(rng, pb["consume_b"]).format(**ctx),
        pick(rng, pb["sell"]).format(**ctx),
        pick(rng, pb["question"]).format(**ctx),

    ]

    question = build_question_with_optional_skin(parts, pb, seed.template, rng)

    answer = " ".join([
        pick(rng, pb["reason_intro"]).format(**ctx),
        pick(rng, pb["reason_use"]).format(**ctx),
        pick(rng, pb["reason_calc"]).format(**ctx),
        pick(rng, pb["reason_final"]).format(**ctx),
    ])

    return f"{question}\nCevap: {answer}"


def render_remainder_after_loss(seed: RemainderAfterLossSeed, pb: Dict[str, Any], rng: random.Random) -> str:
    ctx = seed_to_ctx(seed, rng)
    ctx["start"] = pick_int(rng, seed.start)
    ctx["lost"] = pick_int(rng, seed.lost)

    remain = ctx["start"] - ctx["lost"]
    ctx["remain"] = remain

    parts = [
        pick(rng, pb["loss_story_start"]).format(**ctx),
        pick(rng, pb["loss_story_event"]).format(**ctx),
        pick(rng, pb["loss_question"]).format(**ctx),

    ]

    question = build_question_with_optional_skin(parts, pb, seed.template, rng)

    answer = " ".join([
        pick(rng, pb["loss_reason_intro"]).format(**ctx),
        pick(rng, pb["loss_reason_calc"]).format(**ctx),
        pick(rng, pb["loss_reason_final"]).format(**ctx),
    ])

    return f"{question}\nCevap: {answer}"


def render_equal_sharing(seed: EqualSharingSeed, pb: Dict[str, Any], rng: random.Random) -> str:
    ctx = seed_to_ctx(seed, rng)
    ctx["total"] = pick_int(rng, seed.total)
    ctx["people"] = pick_int(rng, seed.people)

    each = ctx["total"] // ctx["people"]
    ctx["each"] = each

    parts = [
        pick(rng, pb["share_story_start"]).format(**ctx),
        pick(rng, pb["share_story_event"]).format(**ctx),
        pick(rng, pb["share_question"]).format(**ctx),

    ]

    question = build_question_with_optional_skin(parts, pb, seed.template, rng)

    answer = " ".join([
        pick(rng, pb["share_reason_intro"]).format(**ctx),
        pick(rng, pb["share_reason_calc"]).format(**ctx),
        pick(rng, pb["share_reason_final"]).format(**ctx),
    ])

    return f"{question}\nCevap: {answer}"


def render_multi_step_add_sub(seed: MultiStepAddSubSeed, pb: Dict[str, Any], rng: random.Random) -> str:
    ctx = seed_to_ctx(seed, rng)
    ctx["start"] = pick_int(rng, seed.start)
    ctx["add"] = pick_int(rng, seed.add)
    ctx["sub"] = pick_int(rng, seed.sub)
    ctx["start_plus"] = ctx["start"] + ctx["add"]

    final = ctx["start"] + ctx["add"] - ctx["sub"]
    ctx["final"] = final

    parts = [
        pick(rng, pb["multistep_story_start"]).format(**ctx),
        pick(rng, pb["multistep_story_event"]).format(**ctx),
        pick(rng, pb["multistep_question"]).format(**ctx),

    ]

    question = build_question_with_optional_skin(parts, pb, seed.template, rng)

    answer = " ".join([
        pick(rng, pb["multistep_reason_intro"]).format(**ctx),
        pick(rng, pb["multistep_reason_calc"]).format(**ctx),
        pick(rng, pb["multistep_reason_final"]).format(**ctx),
    ])
    return f"{question}\nCevap: {answer}"


def render_unit_price_quantity(seed: UnitPriceQuantitySeed, pb: Dict[str, Any], rng: random.Random) -> str:
    ctx = seed_to_ctx(seed, rng)
    ctx["price"] = pick_int(rng, seed.price)
    ctx["qty"] = pick_int(rng, seed.qty)

    total = ctx["price"] * ctx["qty"]
    ctx["total"] = total

    parts = [
        pick(rng, pb["unit_story_start"]).format(**ctx),
        pick(rng, pb["unit_story_event"]).format(**ctx),
        pick(rng, pb["unit_question"]).format(**ctx),

    ]

    question = build_question_with_optional_skin(parts, pb, seed.template, rng)
    answer = " ".join([
        pick(rng, pb["unit_reason_intro"]).format(**ctx),
        pick(rng, pb["unit_reason_calc"]).format(**ctx),
        pick(rng, pb["unit_reason_final"]).format(**ctx),
    ])
    return f"{question}\nCevap: {answer}"


def render_rate_time(seed: RateTimeSeed, pb: Dict[str, Any], rng: random.Random) -> str:
    ctx = seed_to_ctx(seed, rng)
    ctx["rate"] = pick_int(rng, seed.rate)
    ctx["time"] = pick_int(rng, seed.time)
    ctx["time_unit"] = "dakika"

    total = ctx["rate"] * ctx["time"]
    ctx["total"] = total

    parts = [
        pick(rng, pb["rate_story_start"]).format(**ctx),
        pick(rng, pb["rate_story_event"]).format(**ctx),
        pick(rng, pb["rate_question"]).format(**ctx),

    ]

    question = build_question_with_optional_skin(parts, pb, seed.template, rng)
    answer = " ".join([
        pick(rng, pb["rate_reason_intro"]).format(**ctx),
        pick(rng, pb["rate_reason_calc"]).format(**ctx),
        pick(rng, pb["rate_reason_final"]).format(**ctx),
    ])
    return f"{question}\nCevap: {answer}"


def render_ratio_scaling(seed: RatioScalingSeed, pb: Dict[str, Any], rng: random.Random) -> str:
    ctx = seed_to_ctx(seed, rng)
    ctx["ratio_a"] = pick_int(rng, seed.ratio_a)
    ctx["ratio_b"] = pick_int(rng, seed.ratio_b)
    ctx["total"] = pick_int(rng, seed.total)
    ctx["ask_side"] = seed.ask_side

    unit = ctx["total"] // (ctx["ratio_a"] + ctx["ratio_b"])
    a_count = ctx["ratio_a"] * unit
    b_count = ctx["ratio_b"] * unit
    ctx["unit"] = unit
    ctx["a_count"] = a_count
    ctx["b_count"] = b_count

    if ctx["ask_side"] == "a":
        ctx["asked_item"] = ctx["item_a"]
        ctx["asked_count"] = ctx["a_count"]
    else:
        ctx["asked_item"] = ctx["item_b"]
        ctx["asked_count"] = ctx["b_count"]

    parts = [
        pick(rng, pb["ratio_story_start"]).format(**ctx),
        pick(rng, pb["ratio_story_event"]).format(**ctx),
        pick(rng, pb["ratio_question"]).format(**ctx),

    ]

    question = build_question_with_optional_skin(parts, pb, seed.template, rng)
    answer = " ".join([
        pick(rng, pb["ratio_reason_intro"]).format(**ctx),
        pick(rng, pb["ratio_reason_calc"]).format(**ctx),
        pick(rng, pb["ratio_reason_final"]).format(**ctx),
    ])
    return f"{question}\nCevap: {answer}"


def render_sum_and_difference(seed: SumAndDifferenceSeed, pb: Dict[str, Any], rng: random.Random) -> str:
    ctx = seed_to_ctx(seed, rng)
    ctx["total"] = pick_int(rng, seed.total)
    ctx["diff"] = pick_int(rng, seed.diff)

    big = (ctx["total"] + ctx["diff"]) // 2
    small = ctx["total"] - big
    ctx["big"] = big
    ctx["small"] = small

    parts = [
        pick(rng, pb["sumdiff_story_start"]).format(**ctx),
        pick(rng, pb["sumdiff_story_event"]).format(**ctx),
        pick(rng, pb["sumdiff_question"]).format(**ctx),

    ]

    question = build_question_with_optional_skin(parts, pb, seed.template, rng)
    answer = " ".join([
        pick(rng, pb["sumdiff_reason_intro"]).format(**ctx),
        pick(rng, pb["sumdiff_reason_calc"]).format(**ctx),
        pick(rng, pb["sumdiff_reason_final"]).format(**ctx),
    ])
    return f"{question}\nCevap: {answer}"


def render_compare_difference(seed: CompareDifferenceSeed, pb: Dict[str, Any], rng: random.Random) -> str:
    ctx = seed_to_ctx(seed, rng)
    ctx["item"] = pick_str(rng, seed.item)
    ctx["a"] = pick_int(rng, seed.a)
    ctx["b"] = pick_int(rng, seed.b)

    diff = ctx["a"] - ctx["b"]
    ctx["diff"] = diff

    parts = [
        pick(rng, pb["compare_story_start"]).format(**ctx),
        pick(rng, pb["compare_story_event"]).format(**ctx),
        pick(rng, pb["compare_question"]).format(**ctx),

    ]

    question = build_question_with_optional_skin(parts, pb, seed.template, rng)
    answer = " ".join([
        pick(rng, pb["compare_reason_intro"]).format(**ctx),
        pick(rng, pb["compare_reason_calc"]).format(**ctx),
        pick(rng, pb["compare_reason_final"]).format(**ctx),
    ])
    return f"{question}\nCevap: {answer}"


def render_reverse_operation(seed: ReverseOperationSeed, pb: Dict[str, Any], rng: random.Random) -> str:
    ctx = seed_to_ctx(seed, rng)
    ctx["item"] = pick_str(rng, seed.item)
    ctx["end"] = pick_int(rng, seed.end)
    ctx["add"] = pick_int(rng, seed.add)

    start = ctx["end"] - ctx["add"]
    ctx["start"] = start

    parts = [
        pick(rng, pb["reverse_story_start"]).format(**ctx),
        pick(rng, pb["reverse_story_event"]).format(**ctx),
        pick(rng, pb["reverse_question"]).format(**ctx),

    ]

    question = build_question_with_optional_skin(parts, pb, seed.template, rng)
    answer = " ".join([
        pick(rng, pb["reverse_reason_intro"]).format(**ctx),
        pick(rng, pb["reverse_reason_calc"]).format(**ctx),
        pick(rng, pb["reverse_reason_final"]).format(**ctx),
    ])
    return f"{question}\nCevap: {answer}"


def render_buy_two_items_total_cost(seed: BuyTwoItemsTotalCostSeed, pb: Dict[str, Any], rng: random.Random) -> str:
    ctx = seed_to_ctx(seed, rng)

    ctx["item1"] = pick_str(rng, seed.item1)
    ctx["price1"] = pick_int(rng, seed.price1)
    ctx["qty1"] = pick_int(rng, seed.qty1)

    ctx["item2"] = pick_str(rng, seed.item2)
    ctx["price2"] = pick_int(rng, seed.price2)
    ctx["qty2"] = pick_int(rng, seed.qty2)

    cost1 = ctx["price1"] * ctx["qty1"]
    cost2 = ctx["price2"] * ctx["qty2"]
    total = cost1 + cost2

    ctx["cost1"] = cost1
    ctx["cost2"] = cost2
    ctx["total"] = total

    parts = [
        pick(rng, pb["buy2_story_start"]).format(**ctx),
        pick(rng, pb["buy2_story_event"]).format(**ctx),
        pick(rng, pb["buy2_question"]).format(**ctx),

    ]

    question = build_question_with_optional_skin(parts, pb, seed.template, rng)
    answer = " ".join([
        pick(rng, pb["buy2_reason_intro"]).format(**ctx),
        pick(rng, pb["buy2_reason_calc"]).format(**ctx),
        pick(rng, pb["buy2_reason_final"]).format(**ctx),
    ])
    return f"{question}\nCevap: {answer}"


def render_change_from_payment(seed: ChangeFromPaymentSeed, pb: Dict[str, Any], rng: random.Random) -> str:
    ctx = seed_to_ctx(seed, rng)

    ctx["item1"] = pick_str(rng, seed.item1)
    ctx["price1"] = pick_int(rng, seed.price1)
    ctx["qty1"] = pick_int(rng, seed.qty1)

    ctx["item2"] = pick_str(rng, seed.item2)
    ctx["price2"] = pick_int(rng, seed.price2)
    ctx["qty2"] = pick_int(rng, seed.qty2)

    ctx["paid"] = pick_int(rng, seed.paid)

    cost1 = ctx["price1"] * ctx["qty1"]
    cost2 = ctx["price2"] * ctx["qty2"]
    total_cost = cost1 + cost2
    change = ctx["paid"] - total_cost

    ctx["cost1"] = cost1
    ctx["cost2"] = cost2
    ctx["total_cost"] = total_cost
    ctx["change"] = change

    parts = [
        pick(rng, pb["change_story_start"]).format(**ctx),
        pick(rng, pb["change_story_event"]).format(**ctx),
        pick(rng, pb["change_question"]).format(**ctx),

    ]

    question = build_question_with_optional_skin(parts, pb, seed.template, rng)
    answer = " ".join([
        pick(rng, pb["change_reason_intro"]).format(**ctx),
        pick(rng, pb["change_reason_calc"]).format(**ctx),
        pick(rng, pb["change_reason_final"]).format(**ctx),
    ])
    return f"{question}\nCevap: {answer}"


def render_add_then_share(seed: AddThenShareSeed, pb: Dict[str, Any], rng: random.Random) -> str:
    ctx = seed_to_ctx(seed, rng)

    ctx["item"] = pick_str(rng, seed.item)
    ctx["start"] = pick_int(rng, seed.start)
    ctx["add"] = pick_int(rng, seed.add)
    ctx["people"] = pick_int(rng, seed.people)

    total = ctx["start"] + ctx["add"]
    each = total // ctx["people"]

    ctx["total"] = total
    ctx["each"] = each

    parts = [
        pick(rng, pb["addshare_story_start"]).format(**ctx),
        pick(rng, pb["addshare_story_event"]).format(**ctx),
        pick(rng, pb["addshare_question"]).format(**ctx),

    ]

    question = build_question_with_optional_skin(parts, pb, seed.template, rng)
    answer = " ".join([
        pick(rng, pb["addshare_reason_intro"]).format(**ctx),
        pick(rng, pb["addshare_reason_calc"]).format(**ctx),
        pick(rng, pb["addshare_reason_final"]).format(**ctx),
    ])
    return f"{question}\nCevap: {answer}"


def render_percentage_discount_final_price(seed: PercentageDiscountFinalPriceSeed, pb: Dict[str, Any], rng: random.Random) -> str:
    ctx = seed_to_ctx(seed, rng)

    ctx["item"] = pick_str(rng, seed.item)
    ctx["original_price"] = pick_int(rng, seed.original_price)
    ctx["discount_pct"] = pick_int(rng, seed.discount_pct)

    net_pct = 100 - ctx["discount_pct"]
    final_price = (ctx["original_price"] * net_pct) // 100
    discount_amount = ctx["original_price"] - final_price

    ctx["net_pct"] = net_pct
    ctx["discount_amount"] = discount_amount
    ctx["final_price"] = final_price

    parts = [
        pick(rng, pb["discount_story_start"]).format(**ctx),
        pick(rng, pb["discount_story_event"]).format(**ctx),
        pick(rng, pb["discount_question"]).format(**ctx),

    ]

    question = build_question_with_optional_skin(parts, pb, seed.template, rng)
    answer = " ".join([
        pick(rng, pb["discount_reason_intro"]).format(**ctx),
        pick(rng, pb["discount_reason_calc"]).format(**ctx),
        pick(rng, pb["discount_reason_final"]).format(**ctx),
    ])
    return f"{question}\nCevap: {answer}"


def render_percentage_increase_final_price(seed: PercentageIncreaseFinalPriceSeed, pb: Dict[str, Any], rng: random.Random) -> str:
    ctx = seed_to_ctx(seed, rng)

    ctx["item"] = pick_str(rng, seed.item)
    ctx["original_price"] = pick_int(rng, seed.original_price)
    ctx["increase_pct"] = pick_int(rng, seed.increase_pct)

    gross_pct = 100 + ctx["increase_pct"]
    final_price = (ctx["original_price"] * gross_pct) // 100
    increase_amount = final_price - ctx["original_price"]

    ctx["gross_pct"] = gross_pct
    ctx["increase_amount"] = increase_amount
    ctx["final_price"] = final_price

    parts = [
        pick(rng, pb["increase_story_start"]).format(**ctx),
        pick(rng, pb["increase_story_event"]).format(**ctx),
        pick(rng, pb["increase_question"]).format(**ctx),

    ]

    question = build_question_with_optional_skin(parts, pb, seed.template, rng)
    answer = " ".join([
        pick(rng, pb["increase_reason_intro"]).format(**ctx),
        pick(rng, pb["increase_reason_calc"]).format(**ctx),
        pick(rng, pb["increase_reason_final"]).format(**ctx),
    ])
    return f"{question}\nCevap: {answer}"

def render_discounted_unit_total_cost(seed: DiscountedUnitTotalCostSeed, pb: Dict[str, Any], rng: random.Random) -> str:
    ctx = seed_to_ctx(seed, rng)

    ctx["item"] = pick_str(rng, seed.item)
    ctx["unit_price"] = pick_int(rng, seed.unit_price)
    ctx["discount_pct"] = pick_int(rng, seed.discount_pct)
    ctx["qty"] = pick_int(rng, seed.qty)

    net_pct = 100 - ctx["discount_pct"]
    discounted_unit_price = (ctx["unit_price"] * net_pct) // 100
    total = discounted_unit_price * ctx["qty"]
    discount_amount_per_unit = ctx["unit_price"] - discounted_unit_price

    ctx["net_pct"] = net_pct
    ctx["discounted_unit_price"] = discounted_unit_price
    ctx["discount_amount_per_unit"] = discount_amount_per_unit
    ctx["total"] = total

    parts = [
        pick(rng, pb["discunit_story_start"]).format(**ctx),
        pick(rng, pb["discunit_story_event"]).format(**ctx),
        pick(rng, pb["discunit_question"]).format(**ctx),

    ]

    question = build_question_with_optional_skin(parts, pb, seed.template, rng)
    answer = " ".join([
        pick(rng, pb["discunit_reason_intro"]).format(**ctx),
        pick(rng, pb["discunit_reason_calc"]).format(**ctx),
        pick(rng, pb["discunit_reason_final"]).format(**ctx),
    ])
    return f"{question}\nCevap: {answer}"

def render_vat_total_cost(seed: VatTotalCostSeed, pb: Dict[str, Any], rng: random.Random) -> str:
    ctx = seed_to_ctx(seed, rng)

    ctx["item"] = pick_str(rng, seed.item)
    ctx["unit_price"] = pick_int(rng, seed.unit_price)
    ctx["vat_pct"] = pick_int(rng, seed.vat_pct)
    ctx["qty"] = pick_int(rng, seed.qty)

    gross_pct = 100 + ctx["vat_pct"]
    vat_unit_price = (ctx["unit_price"] * gross_pct) // 100
    vat_amount_per_unit = vat_unit_price - ctx["unit_price"]
    total = vat_unit_price * ctx["qty"]

    ctx["gross_pct"] = gross_pct
    ctx["vat_unit_price"] = vat_unit_price
    ctx["vat_amount_per_unit"] = vat_amount_per_unit
    ctx["total"] = total

    parts = [
        pick(rng, pb["vat_story_start"]).format(**ctx),
        pick(rng, pb["vat_story_event"]).format(**ctx),
        pick(rng, pb["vat_question"]).format(**ctx),

    ]

    question = build_question_with_optional_skin(parts, pb, seed.template, rng)
    answer = " ".join([
        pick(rng, pb["vat_reason_intro"]).format(**ctx),
        pick(rng, pb["vat_reason_calc"]).format(**ctx),
        pick(rng, pb["vat_reason_final"]).format(**ctx),
    ])
    return f"{question}\nCevap: {answer}"

def render_bundle_discount_total_cost(seed: BundleDiscountTotalCostSeed, pb: Dict[str, Any], rng: random.Random) -> str:
    ctx = seed_to_ctx(seed, rng)

    ctx["item"] = pick_str(rng, seed.item)
    ctx["unit_price"] = pick_int(rng, seed.unit_price)
    ctx["qty"] = pick_int(rng, seed.qty)
    ctx["discount_pct"] = pick_int(rng, seed.discount_pct)

    gross_total = ctx["unit_price"] * ctx["qty"]
    net_pct = 100 - ctx["discount_pct"]
    final_total = (gross_total * net_pct) // 100
    discount_amount = gross_total - final_total

    ctx["gross_total"] = gross_total
    ctx["net_pct"] = net_pct
    ctx["discount_amount"] = discount_amount
    ctx["final_total"] = final_total

    parts = [
        pick(rng, pb["bundisc_story_start"]).format(**ctx),
        pick(rng, pb["bundisc_story_event"]).format(**ctx),
        pick(rng, pb["bundisc_question"]).format(**ctx),


    ]


    question = build_question_with_optional_skin(parts, pb, seed.template, rng)

    answer = " ".join([
        pick(rng, pb["bundisc_reason_intro"]).format(**ctx),
        pick(rng, pb["bundisc_reason_calc"]).format(**ctx),
        pick(rng, pb["bundisc_reason_final"]).format(**ctx),
    ])
    return f"{question}\nCevap: {answer}"



def render_discount_then_vat_total_cost(seed: DiscountThenVatTotalCostSeed, pb: Dict[str, Any], rng: random.Random, unknown_axis: bool, unknown_prob: float, stats: tuple[Counter, Counter] | None = None) -> str | None:
    ctx = seed_to_ctx(seed, rng)

    ctx["item"] = pick_str(rng, seed.item)
    ctx["unit_price"] = pick_int(rng, seed.unit_price)
    ctx["discount_pct"] = pick_int(rng, seed.discount_pct)
    ctx["vat_pct"] = pick_int(rng, seed.vat_pct)
    ctx["qty"] = pick_int(rng, seed.qty)
    ctx["currency"] = getattr(seed, "currency", "TL")

    net_pct = 100 - ctx["discount_pct"]
    gross_pct = 100 + ctx["vat_pct"]

    discounted_unit = (ctx["unit_price"] * net_pct) // 100
    vat_unit = (discounted_unit * gross_pct) // 100
    total = vat_unit * ctx["qty"]

    ctx["net_pct"] = net_pct
    ctx["gross_pct"] = gross_pct
    ctx["discounted_unit"] = discounted_unit
    ctx["vat_unit"] = vat_unit
    ctx["total"] = total
    ctx["asked_qty"] = ctx["qty"]

    ask_target = choose_tierc_ask_target(seed.template, rng, unknown_axis, unknown_prob)

    # Stats
    if stats is not None:
        template_counter, ask_counter = stats
        template_counter[seed.template] += 1
        ask_counter[(seed.template, ask_target)] += 1

    # Pick story_event/question keys conditioned on ask target
    if ask_target == "qty":
        event_key = "discvat_story_event_qty"
        q_key = "discvat_question_qty"
    else:
        event_key = "discvat_story_event"
        q_key = "discvat_question"

    # Semantic completeness guard: ensure story_event explicitly states all non-asked premises.
    _event_fmt = pick_fmt_with_required(rng, pb.get(event_key, []), tierc_required_event_vars(seed.template, ask_target))
    if _event_fmt is None:
        return None

    parts = [
        pick(rng, pb["discvat_story_start"]).format(**ctx),
        _event_fmt.format(**ctx),
        pick(rng, pb[q_key]).format(**ctx),
    ]
    question = build_question_with_optional_skin(parts, pb, seed.template, rng)

    if ask_target == "qty":
        ans_final_key = "discvat_reason_final_qty"
        ans_calc_key = "discvat_reason_calc_qty"
    else:
        ans_final_key = "discvat_reason_final"
        ans_calc_key = "discvat_reason_calc"

    answer = " ".join([
        pick(rng, pb["discvat_reason_intro"]).format(**ctx),
        pick(rng, pb[ans_calc_key]).format(**ctx),
        pick(rng, pb[ans_final_key]).format(**ctx),
    ])
    return f"{question}\nCevap: {answer}"


def render_two_items_then_bundle_discount_total_cost(seed: TwoItemsThenBundleDiscountTotalCostSeed, pb: Dict[str, Any], rng: random.Random, unknown_axis: bool, unknown_prob: float, stats: tuple[Counter, Counter] | None = None) -> str | None:
    ctx = seed_to_ctx(seed, rng)

    ctx["item1"] = pick_str(rng, seed.item1)
    ctx["unit_price1"] = pick_int(rng, seed.unit_price1)
    ctx["qty1"] = pick_int(rng, seed.qty1)

    ctx["item2"] = pick_str(rng, seed.item2)
    ctx["unit_price2"] = pick_int(rng, seed.unit_price2)
    ctx["qty2"] = pick_int(rng, seed.qty2)

    ctx["discount_pct"] = pick_int(rng, seed.discount_pct)
    ctx["currency"] = getattr(seed, "currency", "TL")

    cost1 = ctx["unit_price1"] * ctx["qty1"]
    cost2 = ctx["unit_price2"] * ctx["qty2"]
    subtotal = cost1 + cost2

    net_pct = 100 - ctx["discount_pct"]
    final_total = (subtotal * net_pct) // 100
    discount_amount = subtotal - final_total

    ctx["cost1"] = cost1
    ctx["cost2"] = cost2
    ctx["subtotal"] = subtotal
    ctx["net_pct"] = net_pct
    ctx["final_total"] = final_total
    ctx["discount_amount"] = discount_amount
    ctx["asked_discount_pct"] = ctx["discount_pct"]

    ask_target = choose_tierc_ask_target(seed.template, rng, unknown_axis, unknown_prob)

    # Stats
    if stats is not None:
        template_counter, ask_counter = stats
        template_counter[seed.template] += 1
        ask_counter[(seed.template, ask_target)] += 1

    if ask_target == "discount_pct":
        event_key = "buy2bund_story_event_discount_pct"
        q_key = "buy2bund_question_discount_pct"
    else:
        event_key = "buy2bund_story_event"
        q_key = "buy2bund_question"

    # Semantic completeness guard: ensure story_event explicitly states all non-asked premises.
    _event_fmt = pick_fmt_with_required(rng, pb.get(event_key, []), tierc_required_event_vars(seed.template, ask_target))
    if _event_fmt is None:
        return None

    parts = [
        pick(rng, pb["buy2bund_story_start"]).format(**ctx),
        _event_fmt.format(**ctx),
        pick(rng, pb[q_key]).format(**ctx),
    ]
    question = build_question_with_optional_skin(parts, pb, seed.template, rng)

    if ask_target == "discount_pct":
        ans_calc_key = "buy2bund_reason_calc_discount_pct"
        ans_final_key = "buy2bund_reason_final_discount_pct"
    else:
        ans_calc_key = "buy2bund_reason_calc"
        ans_final_key = "buy2bund_reason_final"

    answer = " ".join([
        pick(rng, pb["buy2bund_reason_intro"]).format(**ctx),
        pick(rng, pb[ans_calc_key]).format(**ctx),
        pick(rng, pb[ans_final_key]).format(**ctx),
    ])
    return f"{question}\nCevap: {answer}"


def render_discounted_total_then_change(seed: DiscountedTotalThenChangeSeed, pb: Dict[str, Any], rng: random.Random, unknown_axis: bool, unknown_prob: float, stats: tuple[Counter, Counter] | None = None) -> str | None:
    ctx = seed_to_ctx(seed, rng)

    ctx["item"] = pick_str(rng, seed.item)
    ctx["unit_price"] = pick_int(rng, seed.unit_price)
    ctx["qty"] = pick_int(rng, seed.qty)
    ctx["discount_pct"] = pick_int(rng, seed.discount_pct)
    ctx["paid"] = pick_int(rng, seed.paid)
    ctx["currency"] = getattr(seed, "currency", "TL")

    gross_total = ctx["unit_price"] * ctx["qty"]
    net_pct = 100 - ctx["discount_pct"]
    discounted_total = (gross_total * net_pct) // 100
    change = ctx["paid"] - discounted_total

    ctx["gross_total"] = gross_total
    ctx["net_pct"] = net_pct
    ctx["discounted_total"] = discounted_total
    ctx["change"] = change
    ctx["asked_paid"] = ctx["paid"]

    ask_target = choose_tierc_ask_target(seed.template, rng, unknown_axis, unknown_prob)

    # Stats
    if stats is not None:
        template_counter, ask_counter = stats
        template_counter[seed.template] += 1
        ask_counter[(seed.template, ask_target)] += 1

    if ask_target == "paid":
        event_key = "discpay_story_event_paid"
        q_key = "discpay_question_paid"
    else:
        event_key = "discpay_story_event"
        q_key = "discpay_question"

    # Semantic completeness guard: ensure story_event explicitly states all non-asked premises.
    _event_fmt = pick_fmt_with_required(rng, pb.get(event_key, []), tierc_required_event_vars(seed.template, ask_target))
    if _event_fmt is None:
        return None

    parts = [
        pick(rng, pb["discpay_story_start"]).format(**ctx),
        _event_fmt.format(**ctx),
        pick(rng, pb[q_key]).format(**ctx),
    ]
    question = build_question_with_optional_skin(parts, pb, seed.template, rng)

    if ask_target == "paid":
        ans_calc_key = "discpay_reason_calc_paid"
        ans_final_key = "discpay_reason_final_paid"
    else:
        ans_calc_key = "discpay_reason_calc"
        ans_final_key = "discpay_reason_final"

    answer = " ".join([
        pick(rng, pb["discpay_reason_intro"]).format(**ctx),
        pick(rng, pb[ans_calc_key]).format(**ctx),
        pick(rng, pb[ans_final_key]).format(**ctx),
    ])
    return f"{question}\nCevap: {answer}"


def render_transform_then_share(seed: TransformThenShareSeed, pb: Dict[str, Any], rng: random.Random, unknown_axis: bool, unknown_prob: float, stats: tuple[Counter, Counter] | None = None) -> str | None:
    ctx = seed_to_ctx(seed, rng)

    ctx["item"] = pick_str(rng, seed.item)
    ctx["unit_price"] = pick_int(rng, seed.unit_price)
    ctx["qty"] = pick_int(rng, seed.qty)
    ctx["discount_pct"] = pick_int(rng, seed.discount_pct)
    ctx["people"] = pick_int(rng, seed.people)
    ctx["currency"] = getattr(seed, "currency", "TL")

    gross_total = ctx["unit_price"] * ctx["qty"]
    net_pct = 100 - ctx["discount_pct"]
    discounted_total = (gross_total * net_pct) // 100
    each = discounted_total // ctx["people"]

    ctx["gross_total"] = gross_total
    ctx["net_pct"] = net_pct
    ctx["discounted_total"] = discounted_total
    ctx["each"] = each
    ctx["asked_people"] = ctx["people"]

    ask_target = choose_tierc_ask_target(seed.template, rng, unknown_axis, unknown_prob)

    # Stats
    if stats is not None:
        template_counter, ask_counter = stats
        template_counter[seed.template] += 1
        ask_counter[(seed.template, ask_target)] += 1

    if ask_target == "people":
        event_key = "discsplit_story_event_people"
        q_key = "discsplit_question_people"
    else:
        event_key = "discsplit_story_event"
        q_key = "discsplit_question"

    # Semantic completeness guard: ensure story_event explicitly states all non-asked premises.
    _event_fmt = pick_fmt_with_required(rng, pb.get(event_key, []), tierc_required_event_vars(seed.template, ask_target))
    if _event_fmt is None:
        return None

    parts = [
        pick(rng, pb["discsplit_story_start"]).format(**ctx),
        _event_fmt.format(**ctx),
        pick(rng, pb[q_key]).format(**ctx),
    ]
    question = build_question_with_optional_skin(parts, pb, seed.template, rng)

    if ask_target == "people":
        ans_calc_key = "discsplit_reason_calc_people"
        ans_final_key = "discsplit_reason_final_people"
    else:
        ans_calc_key = "discsplit_reason_calc"
        ans_final_key = "discsplit_reason_final"

    answer = " ".join([
        pick(rng, pb["discsplit_reason_intro"]).format(**ctx),
        pick(rng, pb[ans_calc_key]).format(**ctx),
        pick(rng, pb[ans_final_key]).format(**ctx),
    ])
    return f"{question}\nCevap: {answer}"


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--phrase_bank", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--variants_per_seed", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--unknown_axis", action="store_true", help="Enable unknown-position ask-target variants (Tier C only).")
    ap.add_argument("--unknown_prob", type=float, default=0.25, help="When --unknown_axis is enabled, probability of sampling a non-default ask-target for Tier C (0..1).")
    ap.add_argument("--intent_axis", action="store_true", help="Enable intent skins (kampanya/etiket/kasada/fiş) as a prompt-level axis.")
    ap.add_argument("--intent_prob", type=float, default=0.35, help="When --intent_axis is enabled, probability of applying an intent skin per example (0..1).")

    ap.add_argument("--distractor_axis", action="store_true", help="Enable conservative distractor injection as a prompt-level axis.")
    ap.add_argument("--distractor_prob", type=float, default=0.10, help="When --distractor_axis is enabled, probability of injecting a distractor sentence per example (0..1).")

    ap.add_argument("--dump_stats", action="store_true", help="Print template and ask-target frequency statistics after generation.")
    args = ap.parse_args()

    pb = load_json(args.phrase_bank)
    seed_dicts = load_jsonl(args.seeds)

    seeds: List[Seed] = []
    for d in seed_dicts:
        try:
            s = seed_from_dict(d)
            validate_seed(s)
            seeds.append(s)
        except Exception as e:
            print("\nBAD SEED DETECTED:")
            print(d)
            print("ERROR:", e)
            raise

    examples: List[str] = []
    template_counter = Counter()
    ask_counter = Counter()
    intent_hits = 0
    distractor_hits = 0

    for i, seed in enumerate(seeds):
        for v in range(args.variants_per_seed):
            rng = random.Random(args.seed + i * 10007 + v * 1009)

            if isinstance(seed, ProduceConsumeSellSeed):
                txt0 = render_produce_consume_sell(seed, pb, rng)
                txt1, d_applied = apply_distractor_axis(txt0, rng, pb, getattr(seed, "template", None), args.distractor_axis, args.distractor_prob)
                if d_applied:
                    distractor_hits += 1
                txt, applied = apply_intent_skin(txt1, rng, pb, args.intent_axis, args.intent_prob)
                if applied:
                    intent_hits += 1
                examples.append(txt)
            elif isinstance(seed, RemainderAfterLossSeed):
                txt0 = render_remainder_after_loss(seed, pb, rng)
                txt1, d_applied = apply_distractor_axis(txt0, rng, pb, getattr(seed, "template", None), args.distractor_axis, args.distractor_prob)
                if d_applied:
                    distractor_hits += 1
                txt, applied = apply_intent_skin(txt1, rng, pb, args.intent_axis, args.intent_prob)
                if applied:
                    intent_hits += 1
                examples.append(txt)
            elif isinstance(seed, EqualSharingSeed):
                txt0 = render_equal_sharing(seed, pb, rng)
                txt1, d_applied = apply_distractor_axis(txt0, rng, pb, getattr(seed, "template", None), args.distractor_axis, args.distractor_prob)
                if d_applied:
                    distractor_hits += 1
                txt, applied = apply_intent_skin(txt1, rng, pb, args.intent_axis, args.intent_prob)
                if applied:
                    intent_hits += 1
                examples.append(txt)
            elif isinstance(seed, MultiStepAddSubSeed):
                txt0 = render_multi_step_add_sub(seed, pb, rng)
                txt1, d_applied = apply_distractor_axis(txt0, rng, pb, getattr(seed, "template", None), args.distractor_axis, args.distractor_prob)
                if d_applied:
                    distractor_hits += 1
                txt, applied = apply_intent_skin(txt1, rng, pb, args.intent_axis, args.intent_prob)
                if applied:
                    intent_hits += 1
                examples.append(txt)
            elif isinstance(seed, UnitPriceQuantitySeed):
                txt0 = render_unit_price_quantity(seed, pb, rng)
                txt1, d_applied = apply_distractor_axis(txt0, rng, pb, getattr(seed, "template", None), args.distractor_axis, args.distractor_prob)
                if d_applied:
                    distractor_hits += 1
                txt, applied = apply_intent_skin(txt1, rng, pb, args.intent_axis, args.intent_prob)
                if applied:
                    intent_hits += 1
                examples.append(txt)
            elif isinstance(seed, RateTimeSeed):
                txt0 = render_rate_time(seed, pb, rng)
                txt1, d_applied = apply_distractor_axis(txt0, rng, pb, getattr(seed, "template", None), args.distractor_axis, args.distractor_prob)
                if d_applied:
                    distractor_hits += 1
                txt, applied = apply_intent_skin(txt1, rng, pb, args.intent_axis, args.intent_prob)
                if applied:
                    intent_hits += 1
                examples.append(txt)
            elif isinstance(seed, RatioScalingSeed):
                txt0 = render_ratio_scaling(seed, pb, rng)
                txt1, d_applied = apply_distractor_axis(txt0, rng, pb, getattr(seed, "template", None), args.distractor_axis, args.distractor_prob)
                if d_applied:
                    distractor_hits += 1
                txt, applied = apply_intent_skin(txt1, rng, pb, args.intent_axis, args.intent_prob)
                if applied:
                    intent_hits += 1
                examples.append(txt)
            elif isinstance(seed, SumAndDifferenceSeed):
                txt0 = render_sum_and_difference(seed, pb, rng)
                txt1, d_applied = apply_distractor_axis(txt0, rng, pb, getattr(seed, "template", None), args.distractor_axis, args.distractor_prob)
                if d_applied:
                    distractor_hits += 1
                txt, applied = apply_intent_skin(txt1, rng, pb, args.intent_axis, args.intent_prob)
                if applied:
                    intent_hits += 1
                examples.append(txt)
            elif isinstance(seed, CompareDifferenceSeed):
                txt0 = render_compare_difference(seed, pb, rng)
                txt1, d_applied = apply_distractor_axis(txt0, rng, pb, getattr(seed, "template", None), args.distractor_axis, args.distractor_prob)
                if d_applied:
                    distractor_hits += 1
                txt, applied = apply_intent_skin(txt1, rng, pb, args.intent_axis, args.intent_prob)
                if applied:
                    intent_hits += 1
                examples.append(txt)
            elif isinstance(seed, ReverseOperationSeed):
                txt0 = render_reverse_operation(seed, pb, rng)
                txt1, d_applied = apply_distractor_axis(txt0, rng, pb, getattr(seed, "template", None), args.distractor_axis, args.distractor_prob)
                if d_applied:
                    distractor_hits += 1
                txt, applied = apply_intent_skin(txt1, rng, pb, args.intent_axis, args.intent_prob)
                if applied:
                    intent_hits += 1
                examples.append(txt)
            elif isinstance(seed, BuyTwoItemsTotalCostSeed):
                txt0 = render_buy_two_items_total_cost(seed, pb, rng)
                txt1, d_applied = apply_distractor_axis(txt0, rng, pb, getattr(seed, "template", None), args.distractor_axis, args.distractor_prob)
                if d_applied:
                    distractor_hits += 1
                txt, applied = apply_intent_skin(txt1, rng, pb, args.intent_axis, args.intent_prob)
                if applied:
                    intent_hits += 1
                examples.append(txt)
            elif isinstance(seed, ChangeFromPaymentSeed):
                txt0 = render_change_from_payment(seed, pb, rng)
                txt1, d_applied = apply_distractor_axis(txt0, rng, pb, getattr(seed, "template", None), args.distractor_axis, args.distractor_prob)
                if d_applied:
                    distractor_hits += 1
                txt, applied = apply_intent_skin(txt1, rng, pb, args.intent_axis, args.intent_prob)
                if applied:
                    intent_hits += 1
                examples.append(txt)
            elif isinstance(seed, AddThenShareSeed):
                txt0 = render_add_then_share(seed, pb, rng)
                txt1, d_applied = apply_distractor_axis(txt0, rng, pb, getattr(seed, "template", None), args.distractor_axis, args.distractor_prob)
                if d_applied:
                    distractor_hits += 1
                txt, applied = apply_intent_skin(txt1, rng, pb, args.intent_axis, args.intent_prob)
                if applied:
                    intent_hits += 1
                examples.append(txt)
            elif isinstance(seed, PercentageDiscountFinalPriceSeed):
                txt0 = render_percentage_discount_final_price(seed, pb, rng)
                txt1, d_applied = apply_distractor_axis(txt0, rng, pb, getattr(seed, "template", None), args.distractor_axis, args.distractor_prob)
                if d_applied:
                    distractor_hits += 1
                txt, applied = apply_intent_skin(txt1, rng, pb, args.intent_axis, args.intent_prob)
                if applied:
                    intent_hits += 1
                examples.append(txt)
            elif isinstance(seed, PercentageIncreaseFinalPriceSeed):
                txt0 = render_percentage_increase_final_price(seed, pb, rng)
                txt1, d_applied = apply_distractor_axis(txt0, rng, pb, getattr(seed, "template", None), args.distractor_axis, args.distractor_prob)
                if d_applied:
                    distractor_hits += 1
                txt, applied = apply_intent_skin(txt1, rng, pb, args.intent_axis, args.intent_prob)
                if applied:
                    intent_hits += 1
                examples.append(txt)
            elif isinstance(seed, DiscountedUnitTotalCostSeed):
                txt0 = render_discounted_unit_total_cost(seed, pb, rng)
                txt1, d_applied = apply_distractor_axis(txt0, rng, pb, getattr(seed, "template", None), args.distractor_axis, args.distractor_prob)
                if d_applied:
                    distractor_hits += 1
                txt, applied = apply_intent_skin(txt1, rng, pb, args.intent_axis, args.intent_prob)
                if applied:
                    intent_hits += 1
                examples.append(txt)
            elif isinstance(seed, VatTotalCostSeed):
                txt0 = render_vat_total_cost(seed, pb, rng)
                txt1, d_applied = apply_distractor_axis(txt0, rng, pb, getattr(seed, "template", None), args.distractor_axis, args.distractor_prob)
                if d_applied:
                    distractor_hits += 1
                txt, applied = apply_intent_skin(txt1, rng, pb, args.intent_axis, args.intent_prob)
                if applied:
                    intent_hits += 1
                examples.append(txt)
            elif isinstance(seed, BundleDiscountTotalCostSeed):
                txt0 = render_bundle_discount_total_cost(seed, pb, rng)
                txt1, d_applied = apply_distractor_axis(txt0, rng, pb, getattr(seed, "template", None), args.distractor_axis, args.distractor_prob)
                if d_applied:
                    distractor_hits += 1
                txt, applied = apply_intent_skin(txt1, rng, pb, args.intent_axis, args.intent_prob)
                if applied:
                    intent_hits += 1
                examples.append(txt)
            elif isinstance(seed, DiscountThenVatTotalCostSeed):
                out = render_discount_then_vat_total_cost(seed, pb, rng, args.unknown_axis, args.unknown_prob, stats=(template_counter, ask_counter))
                if out is not None:
                    txt1, d_applied = apply_distractor_axis(out, rng, pb, getattr(seed, "template", None), args.distractor_axis, args.distractor_prob)
                    if d_applied:
                        distractor_hits += 1
                    txt, applied = apply_intent_skin(txt1, rng, pb, args.intent_axis, args.intent_prob)
                    if applied:
                        intent_hits += 1
                    examples.append(txt)
            elif isinstance(seed, TwoItemsThenBundleDiscountTotalCostSeed):
                out = render_two_items_then_bundle_discount_total_cost(seed, pb, rng, args.unknown_axis, args.unknown_prob, stats=(template_counter, ask_counter))
                if out is not None:
                    txt1, d_applied = apply_distractor_axis(out, rng, pb, getattr(seed, "template", None), args.distractor_axis, args.distractor_prob)
                    if d_applied:
                        distractor_hits += 1
                    txt, applied = apply_intent_skin(txt1, rng, pb, args.intent_axis, args.intent_prob)
                    if applied:
                        intent_hits += 1
                    examples.append(txt)
            elif isinstance(seed, DiscountedTotalThenChangeSeed):
                out = render_discounted_total_then_change(seed, pb, rng, args.unknown_axis, args.unknown_prob, stats=(template_counter, ask_counter))
                if out is not None:
                    txt1, d_applied = apply_distractor_axis(out, rng, pb, getattr(seed, "template", None), args.distractor_axis, args.distractor_prob)
                    if d_applied:
                        distractor_hits += 1
                    txt, applied = apply_intent_skin(txt1, rng, pb, args.intent_axis, args.intent_prob)
                    if applied:
                        intent_hits += 1
                    examples.append(txt)
            elif isinstance(seed, TransformThenShareSeed):
                out = render_transform_then_share(seed, pb, rng, args.unknown_axis, args.unknown_prob, stats=(template_counter, ask_counter))
                if out is not None:
                    txt1, d_applied = apply_distractor_axis(out, rng, pb, getattr(seed, "template", None), args.distractor_axis, args.distractor_prob)
                    if d_applied:
                        distractor_hits += 1
                    txt, applied = apply_intent_skin(txt1, rng, pb, args.intent_axis, args.intent_prob)
                    if applied:
                        intent_hits += 1
                    examples.append(txt)
            else:
                raise ValueError(f"Unhandled seed type: {seed}")

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n\n".join(examples))


    if args.dump_stats:
        print("\n=== TEMPLATE COUNTS ===")
        for k, v in template_counter.most_common():
            print(k, v)

        print("\n=== ASK-TARGET COUNTS ===")
        for (tpl, ask), v in ask_counter.most_common():
            print(tpl, ask, v)


            total = len(examples)

        print("\n=== DISTRACTOR HIT RATE ===")
        if total == 0:
            print("distractor_applied 0/0 (0.0%)")
        else:
            pct = (distractor_hits / total) * 100.0
            print(f"distractor_applied {distractor_hits}/{total} ({pct:.1f}%)")

        print("\n=== INTENT HIT RATE ===")
        if total == 0:
            print("intent_applied 0/0 (0.0%)")
        else:
            pct = (intent_hits / total) * 100.0
            print(f"intent_applied {intent_hits}/{total} ({pct:.1f}%)")



if __name__ == "__main__":
    main()