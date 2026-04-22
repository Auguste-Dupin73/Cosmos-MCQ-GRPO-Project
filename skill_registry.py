"""Template-to-skill mapping for the episode pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from utils_episode import ensure_legacy_paths

ensure_legacy_paths()

from schemas_seed import _TEMPLATE_MAP  # type: ignore[attr-defined]


@dataclass(frozen=True)
class SkillSpec:
    skill_id: str
    template_id: str
    tier: str
    default_difficulty: str
    topic: str
    operation_chain: List[str]
    common_error_types: List[str]
    compatible_probe_templates: List[str]


def _difficulty_for_tier(tier: str) -> str:
    return {"A": "easy", "B": "medium", "C": "hard"}[tier]


_COMMON_ERRORS = [
    "near_miss_small_offset",
    "copied_operand",
    "omitted_step",
    "wrong_operation_order",
    "correct_option_wrong_reasoning",
]


def _spec(
    *,
    skill_id: str,
    template_id: str,
    tier: str,
    topic: str,
    operation_chain: List[str],
    compatible_probe_templates: List[str] | None = None,
) -> SkillSpec:
    return SkillSpec(
        skill_id=skill_id,
        template_id=template_id,
        tier=tier,
        default_difficulty=_difficulty_for_tier(tier),
        topic=topic,
        operation_chain=operation_chain,
        common_error_types=list(_COMMON_ERRORS),
        compatible_probe_templates=compatible_probe_templates or [template_id],
    )


_PAIR_PRODUCE = ["produce_consume_sell", "sell_leftover"]

TEMPLATE_SPECS: Dict[str, SkillSpec] = {
    "produce_consume_sell": _spec(
        skill_id="net_remainder_revenue",
        template_id="produce_consume_sell",
        tier="A",
        topic="günlük üretim ve satış",
        operation_chain=["subtract", "subtract", "multiply"],
        compatible_probe_templates=_PAIR_PRODUCE,
    ),
    "sell_leftover": _spec(
        skill_id="net_remainder_revenue",
        template_id="sell_leftover",
        tier="A",
        topic="günlük üretim ve satış",
        operation_chain=["subtract", "subtract", "multiply"],
        compatible_probe_templates=_PAIR_PRODUCE,
    ),
    "remainder_after_loss": _spec(
        skill_id="remainder_after_loss",
        template_id="remainder_after_loss",
        tier="A",
        topic="eksiltme",
        operation_chain=["subtract"],
    ),
    "equal_sharing": _spec(
        skill_id="equal_sharing",
        template_id="equal_sharing",
        tier="A",
        topic="eşit paylaşım",
        operation_chain=["divide"],
    ),
    "multi_step_add_sub": _spec(
        skill_id="multi_step_add_sub",
        template_id="multi_step_add_sub",
        tier="A",
        topic="çok adımlı toplama çıkarma",
        operation_chain=["add", "subtract"],
    ),
    "unit_price_quantity": _spec(
        skill_id="unit_price_quantity",
        template_id="unit_price_quantity",
        tier="A",
        topic="birim fiyat",
        operation_chain=["multiply"],
    ),
    "rate_time": _spec(
        skill_id="rate_time",
        template_id="rate_time",
        tier="A",
        topic="oran ve süre",
        operation_chain=["multiply"],
    ),
    "ratio_scaling": _spec(
        skill_id="ratio_scaling",
        template_id="ratio_scaling",
        tier="A",
        topic="oran",
        operation_chain=["add", "divide", "multiply"],
    ),
    "sum_and_difference": _spec(
        skill_id="sum_and_difference",
        template_id="sum_and_difference",
        tier="A",
        topic="toplam ve fark",
        operation_chain=["add", "divide"],
    ),
    "compare_difference": _spec(
        skill_id="compare_difference",
        template_id="compare_difference",
        tier="A",
        topic="karşılaştırma",
        operation_chain=["subtract"],
    ),
    "reverse_operation": _spec(
        skill_id="reverse_operation",
        template_id="reverse_operation",
        tier="A",
        topic="ters işlem",
        operation_chain=["subtract"],
    ),
    "buy_two_items_total_cost": _spec(
        skill_id="basket_total",
        template_id="buy_two_items_total_cost",
        tier="B",
        topic="sepet toplamı",
        operation_chain=["multiply", "multiply", "add"],
    ),
    "change_from_payment": _spec(
        skill_id="change_from_payment",
        template_id="change_from_payment",
        tier="B",
        topic="para üstü",
        operation_chain=["multiply", "multiply", "add", "subtract"],
    ),
    "add_then_share": _spec(
        skill_id="add_then_share",
        template_id="add_then_share",
        tier="B",
        topic="topla sonra paylaş",
        operation_chain=["add", "divide"],
    ),
    "percentage_discount_final_price": _spec(
        skill_id="percentage_discount",
        template_id="percentage_discount_final_price",
        tier="B",
        topic="yüzde indirim",
        operation_chain=["multiply", "divide"],
    ),
    "percentage_increase_final_price": _spec(
        skill_id="percentage_increase",
        template_id="percentage_increase_final_price",
        tier="B",
        topic="yüzde artış",
        operation_chain=["multiply", "divide"],
    ),
    "discounted_unit_total_cost": _spec(
        skill_id="discounted_unit_total",
        template_id="discounted_unit_total_cost",
        tier="B",
        topic="indirimli birim fiyat",
        operation_chain=["multiply", "divide", "multiply"],
    ),
    "vat_total_cost": _spec(
        skill_id="vat_total_cost",
        template_id="vat_total_cost",
        tier="B",
        topic="kdv",
        operation_chain=["multiply", "divide", "multiply"],
    ),
    "bundle_discount_total_cost": _spec(
        skill_id="bundle_discount_total",
        template_id="bundle_discount_total_cost",
        tier="B",
        topic="toplam üzerinden indirim",
        operation_chain=["multiply", "multiply", "divide"],
    ),
    "discount_then_vat_total_cost": _spec(
        skill_id="discount_then_vat",
        template_id="discount_then_vat_total_cost",
        tier="C",
        topic="indirim sonra kdv",
        operation_chain=["multiply", "divide", "multiply", "divide", "multiply"],
    ),
    "two_items_then_bundle_discount_total_cost": _spec(
        skill_id="two_items_bundle_discount",
        template_id="two_items_then_bundle_discount_total_cost",
        tier="C",
        topic="iki ürünlü sepet indirimi",
        operation_chain=["multiply", "multiply", "add", "multiply", "divide"],
    ),
    "discounted_total_then_change": _spec(
        skill_id="discounted_total_then_change",
        template_id="discounted_total_then_change",
        tier="C",
        topic="indirimli toplam ve para üstü",
        operation_chain=["multiply", "multiply", "divide", "subtract"],
    ),
    "transform_then_share": _spec(
        skill_id="transform_then_share",
        template_id="transform_then_share",
        tier="C",
        topic="indirim sonrası paylaşım",
        operation_chain=["multiply", "multiply", "divide", "divide"],
    ),
}


missing_templates = sorted(set(_TEMPLATE_MAP) - set(TEMPLATE_SPECS))
if missing_templates:
    raise RuntimeError(f"skill_registry.py is missing template mappings: {missing_templates}")


def get_skill_spec(template_id: str) -> SkillSpec:
    """Return the registered skill metadata for a template."""
    try:
        return TEMPLATE_SPECS[template_id]
    except KeyError as exc:
        raise KeyError(f"Unsupported template in skill registry: {template_id}") from exc


def list_supported_templates() -> List[str]:
    """List every template supported by the episode pipeline."""
    return sorted(TEMPLATE_SPECS)
