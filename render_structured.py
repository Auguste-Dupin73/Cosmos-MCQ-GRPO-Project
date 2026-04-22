"""Deterministic structured rendering for seed-driven episode generation."""

from __future__ import annotations

from dataclasses import asdict, fields
import random
from typing import Any, Callable, Dict

from utils_episode import ensure_legacy_paths, split_legacy_rendered
from skill_registry import get_skill_spec

ensure_legacy_paths()

from generate_raw import (  # type: ignore[import-not-found]
    render_add_then_share,
    render_bundle_discount_total_cost,
    render_buy_two_items_total_cost,
    render_change_from_payment,
    render_compare_difference,
    render_discount_then_vat_total_cost,
    render_discounted_total_then_change,
    render_discounted_unit_total_cost,
    render_equal_sharing,
    render_multi_step_add_sub,
    render_percentage_discount_final_price,
    render_percentage_increase_final_price,
    render_produce_consume_sell,
    render_rate_time,
    render_ratio_scaling,
    render_remainder_after_loss,
    render_reverse_operation,
    render_sum_and_difference,
    render_transform_then_share,
    render_two_items_then_bundle_discount_total_cost,
    render_unit_price_quantity,
    render_vat_total_cost,
)
from schemas_seed import (  # type: ignore[import-not-found]
    AddThenShareSeed,
    BundleDiscountTotalCostSeed,
    BuyTwoItemsTotalCostSeed,
    ChangeFromPaymentSeed,
    CompareDifferenceSeed,
    DiscountThenVatTotalCostSeed,
    DiscountedTotalThenChangeSeed,
    DiscountedUnitTotalCostSeed,
    EqualSharingSeed,
    MultiStepAddSubSeed,
    PercentageDiscountFinalPriceSeed,
    PercentageIncreaseFinalPriceSeed,
    ProduceConsumeSellSeed,
    RateTimeSeed,
    RatioScalingSeed,
    RemainderAfterLossSeed,
    ReverseOperationSeed,
    Seed,
    SumAndDifferenceSeed,
    TransformThenShareSeed,
    TwoItemsThenBundleDiscountTotalCostSeed,
    UnitPriceQuantitySeed,
    VatTotalCostSeed,
    as_list_int,
    as_list_str,
)


LegacyRenderer = Callable[..., str | None]


def _pick_value(value: Any, rng: random.Random) -> Any:
    if isinstance(value, list):
        if not value:
            raise ValueError("Cannot sample from an empty list in seed payload")
        first = value[0]
        if isinstance(first, str):
            return rng.choice(as_list_str(value))
        return rng.choice(as_list_int(value))
    return value


def concretize_seed(seed: Seed, rng: random.Random) -> Seed:
    """Sample a concrete scalar seed from a list-valued seed definition."""
    payload = asdict(seed)
    concrete: dict[str, Any] = {}
    for field in fields(seed):
        raw = payload[field.name]
        if field.name in {"template", "ask_side", "currency"}:
            concrete[field.name] = raw
        elif field.name == "name" and raw is None:
            concrete[field.name] = None
        else:
            concrete[field.name] = _pick_value(raw, rng)
    return type(seed)(**concrete)


def _render_question(seed: Seed, phrase_bank: Dict[str, Any], rng: random.Random) -> str:
    renderer_map: dict[type[Any], LegacyRenderer] = {
        ProduceConsumeSellSeed: render_produce_consume_sell,
        RemainderAfterLossSeed: render_remainder_after_loss,
        EqualSharingSeed: render_equal_sharing,
        MultiStepAddSubSeed: render_multi_step_add_sub,
        UnitPriceQuantitySeed: render_unit_price_quantity,
        RateTimeSeed: render_rate_time,
        RatioScalingSeed: render_ratio_scaling,
        SumAndDifferenceSeed: render_sum_and_difference,
        CompareDifferenceSeed: render_compare_difference,
        ReverseOperationSeed: render_reverse_operation,
        BuyTwoItemsTotalCostSeed: render_buy_two_items_total_cost,
        ChangeFromPaymentSeed: render_change_from_payment,
        AddThenShareSeed: render_add_then_share,
        PercentageDiscountFinalPriceSeed: render_percentage_discount_final_price,
        PercentageIncreaseFinalPriceSeed: render_percentage_increase_final_price,
        DiscountedUnitTotalCostSeed: render_discounted_unit_total_cost,
        VatTotalCostSeed: render_vat_total_cost,
        BundleDiscountTotalCostSeed: render_bundle_discount_total_cost,
        DiscountThenVatTotalCostSeed: render_discount_then_vat_total_cost,
        TwoItemsThenBundleDiscountTotalCostSeed: render_two_items_then_bundle_discount_total_cost,
        DiscountedTotalThenChangeSeed: render_discounted_total_then_change,
        TransformThenShareSeed: render_transform_then_share,
    }
    renderer = renderer_map.get(type(seed))
    if renderer is None:
        raise ValueError(f"Unsupported template for structured rendering: {seed.template}")

    if isinstance(
        seed,
        (
            DiscountThenVatTotalCostSeed,
            TwoItemsThenBundleDiscountTotalCostSeed,
            DiscountedTotalThenChangeSeed,
            TransformThenShareSeed,
        ),
    ):
        rendered = renderer(seed, phrase_bank, rng, False, 0.0)  # type: ignore[misc]
    else:
        rendered = renderer(seed, phrase_bank, rng)  # type: ignore[misc]

    if rendered is None:
        raise ValueError(f"Legacy renderer returned no text for template: {seed.template}")
    question_text, _ = split_legacy_rendered(rendered)
    return question_text


def _structured_result(
    *,
    seed: Seed,
    question_text: str,
    gold_final_answer: int,
    expected_intermediates: list[dict[str, Any]],
    gold_solution_text: str,
) -> dict[str, Any]:
    spec = get_skill_spec(seed.template)
    return {
        "skill_id": spec.skill_id,
        "template_id": seed.template,
        "tier": spec.tier,
        "difficulty": spec.default_difficulty,
        "topic": spec.topic,
        "question_text": question_text,
        "gold_solution_text": gold_solution_text,
        "gold_final_answer": str(gold_final_answer),
        "expected_intermediates": expected_intermediates,
        "operation_chain": spec.operation_chain,
        "concrete_seed": asdict(seed),
    }


def render_structured_seed(seed: Seed, phrase_bank: Dict[str, Any], rng: random.Random) -> dict[str, Any]:
    """Render a seed into deterministic structured math supervision."""
    concrete_seed = concretize_seed(seed, rng)
    question_text = _render_question(concrete_seed, phrase_bank, rng)

    if isinstance(concrete_seed, ProduceConsumeSellSeed):
        remain = concrete_seed.produce - concrete_seed.use1 - concrete_seed.use2
        total = remain * concrete_seed.price
        return _structured_result(
            seed=concrete_seed,
            question_text=question_text,
            gold_final_answer=total,
            expected_intermediates=[
                {"name": "remaining_after_use", "value": remain},
                {"name": "sales_total", "value": total},
            ],
            gold_solution_text=(
                f"Önce kalan miktarı buluruz: {concrete_seed.produce} - {concrete_seed.use1} - "
                f"{concrete_seed.use2} = {remain}. Sonra satış gelirini hesaplarız: "
                f"{remain} x {concrete_seed.price} = {total}."
            ),
        )

    if isinstance(concrete_seed, RemainderAfterLossSeed):
        remain = concrete_seed.start - concrete_seed.lost
        return _structured_result(
            seed=concrete_seed,
            question_text=question_text,
            gold_final_answer=remain,
            expected_intermediates=[{"name": "remaining", "value": remain}],
            gold_solution_text=f"Kalan miktar {concrete_seed.start} - {concrete_seed.lost} = {remain} olur.",
        )

    if isinstance(concrete_seed, EqualSharingSeed):
        each = concrete_seed.total // concrete_seed.people
        return _structured_result(
            seed=concrete_seed,
            question_text=question_text,
            gold_final_answer=each,
            expected_intermediates=[{"name": "share_per_person", "value": each}],
            gold_solution_text=f"Eşit paylaşım için {concrete_seed.total} / {concrete_seed.people} = {each}.",
        )

    if isinstance(concrete_seed, MultiStepAddSubSeed):
        after_add = concrete_seed.start + concrete_seed.add
        final = after_add - concrete_seed.sub
        return _structured_result(
            seed=concrete_seed,
            question_text=question_text,
            gold_final_answer=final,
            expected_intermediates=[
                {"name": "after_add", "value": after_add},
                {"name": "final", "value": final},
            ],
            gold_solution_text=(
                f"Önce ekleriz: {concrete_seed.start} + {concrete_seed.add} = {after_add}. "
                f"Sonra çıkarırız: {after_add} - {concrete_seed.sub} = {final}."
            ),
        )

    if isinstance(concrete_seed, UnitPriceQuantitySeed):
        total = concrete_seed.price * concrete_seed.qty
        return _structured_result(
            seed=concrete_seed,
            question_text=question_text,
            gold_final_answer=total,
            expected_intermediates=[{"name": "total", "value": total}],
            gold_solution_text=f"Toplam tutar {concrete_seed.price} x {concrete_seed.qty} = {total}.",
        )

    if isinstance(concrete_seed, RateTimeSeed):
        total = concrete_seed.rate * concrete_seed.time
        return _structured_result(
            seed=concrete_seed,
            question_text=question_text,
            gold_final_answer=total,
            expected_intermediates=[{"name": "total", "value": total}],
            gold_solution_text=f"Toplam miktar {concrete_seed.rate} x {concrete_seed.time} = {total}.",
        )

    if isinstance(concrete_seed, RatioScalingSeed):
        total_parts = concrete_seed.ratio_a + concrete_seed.ratio_b
        unit = concrete_seed.total // total_parts
        a_count = concrete_seed.ratio_a * unit
        b_count = concrete_seed.ratio_b * unit
        final = a_count if concrete_seed.ask_side == "a" else b_count
        asked_item = concrete_seed.item_a if concrete_seed.ask_side == "a" else concrete_seed.item_b
        return _structured_result(
            seed=concrete_seed,
            question_text=question_text,
            gold_final_answer=final,
            expected_intermediates=[
                {"name": "total_parts", "value": total_parts},
                {"name": "unit_value", "value": unit},
                {"name": "count_a", "value": a_count},
                {"name": "count_b", "value": b_count},
            ],
            gold_solution_text=(
                f"Toplam oran {concrete_seed.ratio_a} + {concrete_seed.ratio_b} = {total_parts}. "
                f"Bir oran birimi {concrete_seed.total} / {total_parts} = {unit}. "
                f"Buna göre {asked_item} sayısı {final} olur."
            ),
        )

    if isinstance(concrete_seed, SumAndDifferenceSeed):
        big = (concrete_seed.total + concrete_seed.diff) // 2
        small = concrete_seed.total - big
        return _structured_result(
            seed=concrete_seed,
            question_text=question_text,
            gold_final_answer=big,
            expected_intermediates=[
                {"name": "bigger_number", "value": big},
                {"name": "smaller_number", "value": small},
            ],
            gold_solution_text=(
                f"Büyük sayı ({concrete_seed.total} + {concrete_seed.diff}) / 2 = {big}. "
                f"Küçük sayı {concrete_seed.total} - {big} = {small}."
            ),
        )

    if isinstance(concrete_seed, CompareDifferenceSeed):
        diff = concrete_seed.a - concrete_seed.b
        return _structured_result(
            seed=concrete_seed,
            question_text=question_text,
            gold_final_answer=diff,
            expected_intermediates=[{"name": "difference", "value": diff}],
            gold_solution_text=f"Fark {concrete_seed.a} - {concrete_seed.b} = {diff}.",
        )

    if isinstance(concrete_seed, ReverseOperationSeed):
        start = concrete_seed.end - concrete_seed.add
        return _structured_result(
            seed=concrete_seed,
            question_text=question_text,
            gold_final_answer=start,
            expected_intermediates=[{"name": "initial_value", "value": start}],
            gold_solution_text=f"Başlangıç miktarı {concrete_seed.end} - {concrete_seed.add} = {start}.",
        )

    if isinstance(concrete_seed, BuyTwoItemsTotalCostSeed):
        cost1 = concrete_seed.price1 * concrete_seed.qty1
        cost2 = concrete_seed.price2 * concrete_seed.qty2
        total = cost1 + cost2
        return _structured_result(
            seed=concrete_seed,
            question_text=question_text,
            gold_final_answer=total,
            expected_intermediates=[
                {"name": "first_item_total", "value": cost1},
                {"name": "second_item_total", "value": cost2},
                {"name": "basket_total", "value": total},
            ],
            gold_solution_text=(
                f"İlk ürün için {concrete_seed.price1} x {concrete_seed.qty1} = {cost1}. "
                f"İkinci ürün için {concrete_seed.price2} x {concrete_seed.qty2} = {cost2}. "
                f"Toplam {cost1} + {cost2} = {total}."
            ),
        )

    if isinstance(concrete_seed, ChangeFromPaymentSeed):
        cost1 = concrete_seed.price1 * concrete_seed.qty1
        cost2 = concrete_seed.price2 * concrete_seed.qty2
        total_cost = cost1 + cost2
        change = concrete_seed.paid - total_cost
        return _structured_result(
            seed=concrete_seed,
            question_text=question_text,
            gold_final_answer=change,
            expected_intermediates=[
                {"name": "first_item_total", "value": cost1},
                {"name": "second_item_total", "value": cost2},
                {"name": "basket_total", "value": total_cost},
                {"name": "change", "value": change},
            ],
            gold_solution_text=(
                f"Ürün toplamları {cost1} ve {cost2} olur. Toplam ödeme {cost1} + {cost2} = {total_cost}. "
                f"Para üstü {concrete_seed.paid} - {total_cost} = {change}."
            ),
        )

    if isinstance(concrete_seed, AddThenShareSeed):
        total = concrete_seed.start + concrete_seed.add
        each = total // concrete_seed.people
        return _structured_result(
            seed=concrete_seed,
            question_text=question_text,
            gold_final_answer=each,
            expected_intermediates=[
                {"name": "combined_total", "value": total},
                {"name": "share_per_person", "value": each},
            ],
            gold_solution_text=(
                f"Önce toplarız: {concrete_seed.start} + {concrete_seed.add} = {total}. "
                f"Sonra paylaşırız: {total} / {concrete_seed.people} = {each}."
            ),
        )

    if isinstance(concrete_seed, PercentageDiscountFinalPriceSeed):
        net_pct = 100 - concrete_seed.discount_pct
        final_price = concrete_seed.original_price * net_pct // 100
        return _structured_result(
            seed=concrete_seed,
            question_text=question_text,
            gold_final_answer=final_price,
            expected_intermediates=[
                {"name": "net_percent", "value": net_pct},
                {"name": "final_price", "value": final_price},
            ],
            gold_solution_text=(
                f"İndirimden sonra kalan oran %{net_pct} olur. "
                f"{concrete_seed.original_price} x {net_pct} / 100 = {final_price}."
            ),
        )

    if isinstance(concrete_seed, PercentageIncreaseFinalPriceSeed):
        gross_pct = 100 + concrete_seed.increase_pct
        final_price = concrete_seed.original_price * gross_pct // 100
        return _structured_result(
            seed=concrete_seed,
            question_text=question_text,
            gold_final_answer=final_price,
            expected_intermediates=[
                {"name": "gross_percent", "value": gross_pct},
                {"name": "final_price", "value": final_price},
            ],
            gold_solution_text=(
                f"Zamlı oran %{gross_pct} olur. {concrete_seed.original_price} x {gross_pct} / 100 = {final_price}."
            ),
        )

    if isinstance(concrete_seed, DiscountedUnitTotalCostSeed):
        net_pct = 100 - concrete_seed.discount_pct
        discounted_unit = concrete_seed.unit_price * net_pct // 100
        total = discounted_unit * concrete_seed.qty
        return _structured_result(
            seed=concrete_seed,
            question_text=question_text,
            gold_final_answer=total,
            expected_intermediates=[
                {"name": "discounted_unit_price", "value": discounted_unit},
                {"name": "total", "value": total},
            ],
            gold_solution_text=(
                f"İndirimli birim fiyat {concrete_seed.unit_price} x {net_pct} / 100 = {discounted_unit}. "
                f"Toplam {discounted_unit} x {concrete_seed.qty} = {total}."
            ),
        )

    if isinstance(concrete_seed, VatTotalCostSeed):
        gross_pct = 100 + concrete_seed.vat_pct
        vat_unit = concrete_seed.unit_price * gross_pct // 100
        total = vat_unit * concrete_seed.qty
        return _structured_result(
            seed=concrete_seed,
            question_text=question_text,
            gold_final_answer=total,
            expected_intermediates=[
                {"name": "vat_unit_price", "value": vat_unit},
                {"name": "total", "value": total},
            ],
            gold_solution_text=(
                f"KDV'li birim fiyat {concrete_seed.unit_price} x {gross_pct} / 100 = {vat_unit}. "
                f"Toplam {vat_unit} x {concrete_seed.qty} = {total}."
            ),
        )

    if isinstance(concrete_seed, BundleDiscountTotalCostSeed):
        gross_total = concrete_seed.unit_price * concrete_seed.qty
        net_pct = 100 - concrete_seed.discount_pct
        final_total = gross_total * net_pct // 100
        return _structured_result(
            seed=concrete_seed,
            question_text=question_text,
            gold_final_answer=final_total,
            expected_intermediates=[
                {"name": "gross_total", "value": gross_total},
                {"name": "final_total", "value": final_total},
            ],
            gold_solution_text=(
                f"İndirimsiz toplam {concrete_seed.unit_price} x {concrete_seed.qty} = {gross_total}. "
                f"İndirim sonrası {gross_total} x {net_pct} / 100 = {final_total}."
            ),
        )

    if isinstance(concrete_seed, DiscountThenVatTotalCostSeed):
        net_pct = 100 - concrete_seed.discount_pct
        gross_pct = 100 + concrete_seed.vat_pct
        discounted_unit = concrete_seed.unit_price * net_pct // 100
        vat_unit = discounted_unit * gross_pct // 100
        total = vat_unit * concrete_seed.qty
        return _structured_result(
            seed=concrete_seed,
            question_text=question_text,
            gold_final_answer=total,
            expected_intermediates=[
                {"name": "discounted_unit_price", "value": discounted_unit},
                {"name": "vat_unit_price", "value": vat_unit},
                {"name": "total", "value": total},
            ],
            gold_solution_text=(
                f"Önce indirim uygularız: {concrete_seed.unit_price} x {net_pct} / 100 = {discounted_unit}. "
                f"Sonra KDV ekleriz: {discounted_unit} x {gross_pct} / 100 = {vat_unit}. "
                f"Toplam {vat_unit} x {concrete_seed.qty} = {total}."
            ),
        )

    if isinstance(concrete_seed, TwoItemsThenBundleDiscountTotalCostSeed):
        cost1 = concrete_seed.unit_price1 * concrete_seed.qty1
        cost2 = concrete_seed.unit_price2 * concrete_seed.qty2
        subtotal = cost1 + cost2
        net_pct = 100 - concrete_seed.discount_pct
        final_total = subtotal * net_pct // 100
        return _structured_result(
            seed=concrete_seed,
            question_text=question_text,
            gold_final_answer=final_total,
            expected_intermediates=[
                {"name": "first_item_total", "value": cost1},
                {"name": "second_item_total", "value": cost2},
                {"name": "subtotal", "value": subtotal},
                {"name": "final_total", "value": final_total},
            ],
            gold_solution_text=(
                f"Ürün toplamları {cost1} ve {cost2}. Ara toplam {cost1} + {cost2} = {subtotal}. "
                f"İndirim sonrası {subtotal} x {net_pct} / 100 = {final_total}."
            ),
        )

    if isinstance(concrete_seed, DiscountedTotalThenChangeSeed):
        gross_total = concrete_seed.unit_price * concrete_seed.qty
        net_pct = 100 - concrete_seed.discount_pct
        discounted_total = gross_total * net_pct // 100
        change = concrete_seed.paid - discounted_total
        return _structured_result(
            seed=concrete_seed,
            question_text=question_text,
            gold_final_answer=change,
            expected_intermediates=[
                {"name": "gross_total", "value": gross_total},
                {"name": "discounted_total", "value": discounted_total},
                {"name": "change", "value": change},
            ],
            gold_solution_text=(
                f"İndirimsiz toplam {concrete_seed.unit_price} x {concrete_seed.qty} = {gross_total}. "
                f"İndirimli toplam {gross_total} x {net_pct} / 100 = {discounted_total}. "
                f"Para üstü {concrete_seed.paid} - {discounted_total} = {change}."
            ),
        )

    if isinstance(concrete_seed, TransformThenShareSeed):
        gross_total = concrete_seed.unit_price * concrete_seed.qty
        net_pct = 100 - concrete_seed.discount_pct
        discounted_total = gross_total * net_pct // 100
        each = discounted_total // concrete_seed.people
        return _structured_result(
            seed=concrete_seed,
            question_text=question_text,
            gold_final_answer=each,
            expected_intermediates=[
                {"name": "gross_total", "value": gross_total},
                {"name": "discounted_total", "value": discounted_total},
                {"name": "share_per_person", "value": each},
            ],
            gold_solution_text=(
                f"İndirimsiz toplam {concrete_seed.unit_price} x {concrete_seed.qty} = {gross_total}. "
                f"İndirim sonrası {gross_total} x {net_pct} / 100 = {discounted_total}. "
                f"Kişi başı {discounted_total} / {concrete_seed.people} = {each}."
            ),
        )

    raise ValueError(f"Unsupported template for structured rendering: {concrete_seed.template}")
