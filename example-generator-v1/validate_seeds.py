from __future__ import annotations

from typing import List, Tuple

from schemas_seed import (
    Seed,
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
    TransformThenShareSeed,
    as_list_int,
    as_list_str,
)


def _minmax(xs: List[int]) -> Tuple[int, int]:
    return (min(xs), max(xs))


def validate_seed(seed: Seed) -> None:
    if isinstance(seed, ProduceConsumeSellSeed):
        _validate_produce_consume_sell(seed); return
    if isinstance(seed, RemainderAfterLossSeed):
        _validate_remainder_after_loss(seed); return
    if isinstance(seed, EqualSharingSeed):
        _validate_equal_sharing(seed); return
    if isinstance(seed, MultiStepAddSubSeed):
        _validate_multi_step_add_sub(seed); return
    if isinstance(seed, UnitPriceQuantitySeed):
        _validate_unit_price_quantity(seed); return
    if isinstance(seed, RateTimeSeed):
        _validate_rate_time(seed); return

    if isinstance(seed, RatioScalingSeed):
        _validate_ratio_scaling(seed); return
    if isinstance(seed, SumAndDifferenceSeed):
        _validate_sum_and_difference(seed); return
    if isinstance(seed, CompareDifferenceSeed):
        _validate_compare_difference(seed); return
    if isinstance(seed, ReverseOperationSeed):
        _validate_reverse_operation(seed); return

    if isinstance(seed, BuyTwoItemsTotalCostSeed):
        _validate_buy_two_items_total_cost(seed); return
    if isinstance(seed, ChangeFromPaymentSeed):
        _validate_change_from_payment(seed); return
    if isinstance(seed, AddThenShareSeed):
        _validate_add_then_share(seed); return

    if isinstance(seed, PercentageDiscountFinalPriceSeed):
        _validate_percentage_discount_final_price(seed); return
    if isinstance(seed, PercentageIncreaseFinalPriceSeed):
        _validate_percentage_increase_final_price(seed); return
    if isinstance(seed, DiscountedUnitTotalCostSeed):
        _validate_discounted_unit_total_cost(seed);
        return
    if isinstance(seed, VatTotalCostSeed):
        _validate_vat_total_cost(seed); return

    if isinstance(seed, BundleDiscountTotalCostSeed):
        _validate_bundle_discount_total_cost(seed);
        return


    if isinstance(seed, DiscountThenVatTotalCostSeed):
        _validate_discount_then_vat_total_cost(seed);
        return
    if isinstance(seed, TwoItemsThenBundleDiscountTotalCostSeed):
        _validate_two_items_then_bundle_discount_total_cost(seed);
        return
    if isinstance(seed, DiscountedTotalThenChangeSeed):
        _validate_discounted_total_then_change(seed);
        return
    if isinstance(seed, TransformThenShareSeed):
        _validate_transform_then_share(seed);
        return

    raise ValueError(f"Unhandled seed type for template={seed.template}")



def _validate_discount_then_vat_total_cost(seed: DiscountThenVatTotalCostSeed) -> None:
    _validate_common_name(seed.name)

    unit_prices = as_list_int(seed.unit_price)
    discount_pcts = as_list_int(seed.discount_pct)
    vat_pcts = as_list_int(seed.vat_pct)
    qtys = as_list_int(seed.qty)

    for unit_price in unit_prices:
        if unit_price <= 0:
            raise ValueError("discount_then_vat_total_cost requires unit_price > 0")
    for qty in qtys:
        if qty <= 0:
            raise ValueError("discount_then_vat_total_cost requires qty > 0")
    for d in discount_pcts:
        if d < 0 or d >= 100:
            raise ValueError("discount_then_vat_total_cost requires 0 <= discount_pct < 100")
    for v in vat_pcts:
        if v < 0 or v >= 100:
            raise ValueError("discount_then_vat_total_cost requires 0 <= vat_pct < 100")

    for unit_price in unit_prices:
        for d in discount_pcts:
            net_pct = 100 - d
            if (unit_price * net_pct) % 100 != 0:
                raise ValueError(
                    "discount_then_vat_total_cost requires integer discounted unit price: "
                    "unit_price*(100-discount_pct) must be divisible by 100 for all combinations"
                )
            discounted_unit = (unit_price * net_pct) // 100
            for v in vat_pcts:
                gross_pct = 100 + v
                if (discounted_unit * gross_pct) % 100 != 0:
                    raise ValueError(
                        "discount_then_vat_total_cost requires integer VAT unit price after discount: "
                        "discounted_unit*(100+vat_pct) must be divisible by 100 for all combinations"
                    )


def _validate_two_items_then_bundle_discount_total_cost(seed: TwoItemsThenBundleDiscountTotalCostSeed) -> None:
    _validate_common_name(seed.name)

    p1s = as_list_int(seed.unit_price1)
    q1s = as_list_int(seed.qty1)
    p2s = as_list_int(seed.unit_price2)
    q2s = as_list_int(seed.qty2)
    dps = as_list_int(seed.discount_pct)

    for p in p1s + p2s:
        if p <= 0:
            raise ValueError("two_items_then_bundle_discount_total_cost requires unit_price > 0")
    for q in q1s + q2s:
        if q <= 0:
            raise ValueError("two_items_then_bundle_discount_total_cost requires qty > 0")
    for d in dps:
        if d < 0 or d >= 100:
            raise ValueError("two_items_then_bundle_discount_total_cost requires 0 <= discount_pct < 100")

    for p1 in p1s:
        for q1 in q1s:
            for p2 in p2s:
                for q2 in q2s:
                    subtotal = p1*q1 + p2*q2
                    for d in dps:
                        net_pct = 100 - d
                        if (subtotal * net_pct) % 100 != 0:
                            raise ValueError(
                                "two_items_then_bundle_discount_total_cost requires integer final total: "
                                "subtotal*(100-discount_pct) must be divisible by 100 for all combinations"
                            )


def _validate_discounted_total_then_change(seed: DiscountedTotalThenChangeSeed) -> None:
    _validate_common_name(seed.name)

    unit_prices = as_list_int(seed.unit_price)
    qtys = as_list_int(seed.qty)
    dps = as_list_int(seed.discount_pct)
    paids = as_list_int(seed.paid)

    for p in unit_prices:
        if p <= 0:
            raise ValueError("discounted_total_then_change requires unit_price > 0")
    for q in qtys:
        if q <= 0:
            raise ValueError("discounted_total_then_change requires qty > 0")
    for d in dps:
        if d < 0 or d >= 100:
            raise ValueError("discounted_total_then_change requires 0 <= discount_pct < 100")
    for paid in paids:
        if paid <= 0:
            raise ValueError("discounted_total_then_change requires paid > 0")

    for p in unit_prices:
        for q in qtys:
            gross_total = p*q
            for d in dps:
                net_pct = 100 - d
                if (gross_total * net_pct) % 100 != 0:
                    raise ValueError(
                        "discounted_total_then_change requires integer discounted total: "
                        "gross_total*(100-discount_pct) must be divisible by 100 for all combinations"
                    )
                discounted_total = (gross_total * net_pct) // 100
                for paid in paids:
                    if paid < discounted_total:
                        raise ValueError(
                            "discounted_total_then_change requires paid >= discounted_total for all combinations"
                        )


def _validate_transform_then_share(seed: TransformThenShareSeed) -> None:
    _validate_common_name(seed.name)

    unit_prices = as_list_int(seed.unit_price)
    qtys = as_list_int(seed.qty)
    dps = as_list_int(seed.discount_pct)
    people = as_list_int(seed.people)

    for p in unit_prices:
        if p <= 0:
            raise ValueError("transform_then_share requires unit_price > 0")
    for q in qtys:
        if q <= 0:
            raise ValueError("transform_then_share requires qty > 0")
    for d in dps:
        if d < 0 or d >= 100:
            raise ValueError("transform_then_share requires 0 <= discount_pct < 100")
    for peep in people:
        if peep <= 0:
            raise ValueError("transform_then_share requires people > 0")

    for p in unit_prices:
        for q in qtys:
            gross_total = p*q
            for d in dps:
                net_pct = 100 - d
                if (gross_total * net_pct) % 100 != 0:
                    raise ValueError(
                        "transform_then_share requires integer discounted total: "
                        "gross_total*(100-discount_pct) must be divisible by 100 for all combinations"
                    )
                discounted_total = (gross_total * net_pct) // 100
                for peep in people:
                    if discounted_total % peep != 0:
                        raise ValueError(
                            "transform_then_share requires discounted_total divisible by people for all combinations"
                        )



def _validate_common_name(name) -> None:
    if name is None:
        return
    names = as_list_str(name)
    if not names:
        raise ValueError("name list cannot be empty")
    for n in names:
        if not str(n).strip():
            raise ValueError("name entries cannot be empty/whitespace")


def _validate_produce_consume_sell(seed: ProduceConsumeSellSeed) -> None:
    _validate_common_name(seed.name)
    if not seed.item or not str(seed.item).strip():
        raise ValueError("item cannot be empty")

    produce = as_list_int(seed.produce)
    use1 = as_list_int(seed.use1)
    use2 = as_list_int(seed.use2)
    price = as_list_int(seed.price)

    if not produce or not use1 or not use2 or not price:
        raise ValueError("numeric lists cannot be empty")

    pmin, _ = _minmax(produce)
    u1min, u1max = _minmax(use1)
    u2min, u2max = _minmax(use2)
    _, pricemax = _minmax(price)

    if pmin <= 0:
        raise ValueError("produce must be positive")
    if u1min < 0 or u2min < 0:
        raise ValueError("use1/use2 cannot be negative")
    if pricemax <= 0:
        raise ValueError("price must be positive")

    if pmin < (u1max + u2max):
        raise ValueError("produce options too small compared to use1/use2 options (could yield negative leftover)")


def _validate_remainder_after_loss(seed: RemainderAfterLossSeed) -> None:
    _validate_common_name(seed.name)
    if not seed.item or not str(seed.item).strip():
        raise ValueError("item cannot be empty")

    start = as_list_int(seed.start)
    lost = as_list_int(seed.lost)
    if not start or not lost:
        raise ValueError("start/lost lists cannot be empty")

    smin, _ = _minmax(start)
    _, lmax = _minmax(lost)

    if smin <= 0:
        raise ValueError("start must be positive")
    if lmax < 0:
        raise ValueError("lost cannot be negative")
    if smin <= lmax:
        raise ValueError("Need min(start) > max(lost) to avoid invalid random draws")


def _validate_equal_sharing(seed: EqualSharingSeed) -> None:
    _validate_common_name(seed.name)
    if not seed.item or not str(seed.item).strip():
        raise ValueError("item cannot be empty")

    total = as_list_int(seed.total)
    people = as_list_int(seed.people)
    if not total or not people:
        raise ValueError("total/people lists cannot be empty")

    if min(total) <= 0:
        raise ValueError("total must be positive")
    if min(people) <= 0:
        raise ValueError("people must be positive")

    for t in total:
        for p in people:
            if p == 0 or t % p != 0:
                raise ValueError("equal_sharing requires every total divisible by every people option")


def _validate_multi_step_add_sub(seed: MultiStepAddSubSeed) -> None:
    _validate_common_name(seed.name)
    if not seed.item or not str(seed.item).strip():
        raise ValueError("item cannot be empty")

    start = as_list_int(seed.start)
    add = as_list_int(seed.add)
    sub = as_list_int(seed.sub)
    if not start or not add or not sub:
        raise ValueError("start/add/sub lists cannot be empty")

    if min(start) < 0 or min(add) < 0:
        raise ValueError("start/add cannot be negative")
    if max(sub) < 0:
        raise ValueError("sub cannot be negative")

    if min(start) + min(add) < max(sub):
        raise ValueError("multi_step_add_sub could go negative for some random draw (min(start)+min(add) < max(sub))")


def _validate_unit_price_quantity(seed: UnitPriceQuantitySeed) -> None:
    _validate_common_name(seed.name)
    if not seed.item or not str(seed.item).strip():
        raise ValueError("item cannot be empty")

    price = as_list_int(seed.price)
    qty = as_list_int(seed.qty)
    if not price or not qty:
        raise ValueError("price/qty lists cannot be empty")

    if min(price) <= 0:
        raise ValueError("price must be positive")
    if min(qty) <= 0:
        raise ValueError("qty must be positive")
    if not seed.currency or not str(seed.currency).strip():
        raise ValueError("currency cannot be empty")


def _validate_rate_time(seed: RateTimeSeed) -> None:
    _validate_common_name(seed.name)
    if not seed.item or not str(seed.item).strip():
        raise ValueError("item cannot be empty")

    rate = as_list_int(seed.rate)
    time = as_list_int(seed.time)
    if not rate or not time:
        raise ValueError("rate/time lists cannot be empty")

    if min(rate) <= 0:
        raise ValueError("rate must be positive")
    if min(time) <= 0:
        raise ValueError("time must be positive")


def _validate_ratio_scaling(seed: RatioScalingSeed) -> None:
    _validate_common_name(seed.name)
    if not seed.item_a.strip() or not seed.item_b.strip():
        raise ValueError("item_a and item_b cannot be empty")

    ra = as_list_int(seed.ratio_a)
    rb = as_list_int(seed.ratio_b)
    total = as_list_int(seed.total)
    if not ra or not rb or not total:
        raise ValueError("ratio_a/ratio_b/total lists cannot be empty")

    if any(x <= 0 for x in ra) or any(x <= 0 for x in rb):
        raise ValueError("ratio parts must be positive")
    if any(t <= 0 for t in total):
        raise ValueError("total must be positive")

    for a in ra:
        for b in rb:
            s = a + b
            for tt in total:
                if tt % s != 0:
                    raise ValueError("ratio_scaling requires every total divisible by every (ratio_a+ratio_b) combination")

    if seed.ask_side not in ("a", "b"):
        raise ValueError("ask_side must be 'a' or 'b'")


def _validate_sum_and_difference(seed: SumAndDifferenceSeed) -> None:
    _validate_common_name(seed.name)

    total = as_list_int(seed.total)
    diff = as_list_int(seed.diff)
    if not total or not diff:
        raise ValueError("total/diff lists cannot be empty")

    if any(t <= 0 for t in total):
        raise ValueError("total must be positive")
    if any(d < 0 for d in diff):
        raise ValueError("diff cannot be negative")

    for t in total:
        for d in diff:
            if d >= t:
                raise ValueError("sum_and_difference requires diff < total for all combinations")
            if (t + d) % 2 != 0:
                raise ValueError("sum_and_difference requires (total+diff) even for all combinations")


def _validate_compare_difference(seed: CompareDifferenceSeed) -> None:
    _validate_common_name(seed.name)

    items = as_list_str(seed.item)
    if not items or any(not it.strip() for it in items):
        raise ValueError("item list cannot be empty and items cannot be blank")

    a = as_list_int(seed.a)
    b = as_list_int(seed.b)
    if not a or not b:
        raise ValueError("a/b lists cannot be empty")

    if min(a) < 0 or min(b) < 0:
        raise ValueError("a and b cannot be negative")
    if min(a) <= max(b):
        raise ValueError("compare_difference requires min(a) > max(b) to avoid negative differences")


def _validate_reverse_operation(seed: ReverseOperationSeed) -> None:
    _validate_common_name(seed.name)

    items = as_list_str(seed.item)
    if not items or any(not it.strip() for it in items):
        raise ValueError("item list cannot be empty and items cannot be blank")

    end = as_list_int(seed.end)
    add = as_list_int(seed.add)
    if not end or not add:
        raise ValueError("end/add lists cannot be empty")

    if min(end) <= 0:
        raise ValueError("end must be positive")
    if max(add) < 0:
        raise ValueError("add cannot be negative")
    if min(end) <= max(add):
        raise ValueError("reverse_operation requires min(end) > max(add) so start is always positive")


# -------------------------
# Phase 3 (Tier B) validators
# -------------------------

def _validate_buy_two_items_total_cost(seed: BuyTwoItemsTotalCostSeed) -> None:
    _validate_common_name(seed.name)

    items1 = as_list_str(seed.item1)
    items2 = as_list_str(seed.item2)
    if not items1 or not items2:
        raise ValueError("item1/item2 lists cannot be empty")
    if any(not it.strip() for it in items1) or any(not it.strip() for it in items2):
        raise ValueError("item1/item2 entries cannot be empty")

    price1 = as_list_int(seed.price1)
    qty1 = as_list_int(seed.qty1)
    price2 = as_list_int(seed.price2)
    qty2 = as_list_int(seed.qty2)

    if not price1 or not qty1 or not price2 or not qty2:
        raise ValueError("price/qty lists cannot be empty")
    if min(price1) <= 0 or min(price2) <= 0:
        raise ValueError("prices must be positive")
    if min(qty1) <= 0 or min(qty2) <= 0:
        raise ValueError("quantities must be positive")
    if not seed.currency or not str(seed.currency).strip():
        raise ValueError("currency cannot be empty")


def _validate_change_from_payment(seed: ChangeFromPaymentSeed) -> None:
    _validate_common_name(seed.name)

    items1 = as_list_str(seed.item1)
    items2 = as_list_str(seed.item2)
    if not items1 or not items2:
        raise ValueError("item1/item2 lists cannot be empty")
    if any(not it.strip() for it in items1) or any(not it.strip() for it in items2):
        raise ValueError("item1/item2 entries cannot be empty")

    price1 = as_list_int(seed.price1)
    qty1 = as_list_int(seed.qty1)
    price2 = as_list_int(seed.price2)
    qty2 = as_list_int(seed.qty2)
    paid = as_list_int(seed.paid)

    if not price1 or not qty1 or not price2 or not qty2 or not paid:
        raise ValueError("numeric lists cannot be empty")

    if min(price1) <= 0 or min(price2) <= 0:
        raise ValueError("prices must be positive")
    if min(qty1) <= 0 or min(qty2) <= 0:
        raise ValueError("quantities must be positive")
    if min(paid) <= 0:
        raise ValueError("paid must be positive")
    if not seed.currency or not str(seed.currency).strip():
        raise ValueError("currency cannot be empty")

    for p1 in price1:
        for q1 in qty1:
            for p2 in price2:
                for q2 in qty2:
                    total_cost = p1 * q1 + p2 * q2
                    for cash in paid:
                        if cash < total_cost:
                            raise ValueError(
                                "change_from_payment requires every paid option >= every possible total cost combination"
                            )


def _validate_add_then_share(seed: AddThenShareSeed) -> None:
    _validate_common_name(seed.name)

    items = as_list_str(seed.item)
    if not items or any(not it.strip() for it in items):
        raise ValueError("item list cannot be empty and items cannot be blank")

    start = as_list_int(seed.start)
    add = as_list_int(seed.add)
    people = as_list_int(seed.people)

    if not start or not add or not people:
        raise ValueError("start/add/people lists cannot be empty")

    if min(start) < 0:
        raise ValueError("start cannot be negative")
    if min(add) < 0:
        raise ValueError("add cannot be negative")
    if min(people) <= 0:
        raise ValueError("people must be positive")

    for s in start:
        for a in add:
            total = s + a
            if total <= 0:
                raise ValueError("add_then_share requires start+add > 0 for all combinations")
            for p in people:
                if total % p != 0:
                    raise ValueError("add_then_share requires (start+add) divisible by every people option for all combinations")


def _validate_percentage_discount_final_price(seed: PercentageDiscountFinalPriceSeed) -> None:
    _validate_common_name(seed.name)

    items = as_list_str(seed.item)
    if not items or any(not it.strip() for it in items):
        raise ValueError("item list cannot be empty and items cannot be blank")

    prices = as_list_int(seed.original_price)
    pcts = as_list_int(seed.discount_pct)

    if not prices or not pcts:
        raise ValueError("original_price/discount_pct lists cannot be empty")

    if min(prices) <= 0:
        raise ValueError("original_price must be positive")
    if any(pct <= 0 or pct >= 100 for pct in pcts):
        raise ValueError("discount_pct must be in 1..99")

    if not seed.currency or not str(seed.currency).strip():
        raise ValueError("currency cannot be empty")

    for price in prices:
        for pct in pcts:
            net = 100 - pct
            if (price * net) % 100 != 0:
                raise ValueError(
                    "percentage_discount_final_price requires integer final price: "
                    "every original_price*(100-discount_pct) must be divisible by 100"
                )


def _validate_percentage_increase_final_price(seed: PercentageIncreaseFinalPriceSeed) -> None:
    _validate_common_name(seed.name)

    items = as_list_str(seed.item)
    if not items or any(not it.strip() for it in items):
        raise ValueError("item list cannot be empty and items cannot be blank")

    prices = as_list_int(seed.original_price)
    pcts = as_list_int(seed.increase_pct)

    if not prices or not pcts:
        raise ValueError("original_price/increase_pct lists cannot be empty")

    if min(prices) <= 0:
        raise ValueError("original_price must be positive")
    if any(pct <= 0 for pct in pcts):
        raise ValueError("increase_pct must be positive")

    if not seed.currency or not str(seed.currency).strip():
        raise ValueError("currency cannot be empty")

    for price in prices:
        for pct in pcts:
            gross = 100 + pct
            if (price * gross) % 100 != 0:
                raise ValueError(
                    "percentage_increase_final_price requires integer final price: "
                    "every original_price*(100+increase_pct) must be divisible by 100"
                )

def _validate_discounted_unit_total_cost(seed: DiscountedUnitTotalCostSeed) -> None:
    _validate_common_name(seed.name)

    items = as_list_str(seed.item)
    if not items or any(not it.strip() for it in items):
        raise ValueError("item list cannot be empty and items cannot be blank")

    unit_prices = as_list_int(seed.unit_price)
    pcts = as_list_int(seed.discount_pct)
    qtys = as_list_int(seed.qty)

    if not unit_prices or not pcts or not qtys:
        raise ValueError("unit_price/discount_pct/qty lists cannot be empty")

    if min(unit_prices) <= 0:
        raise ValueError("unit_price must be positive")
    if any(pct <= 0 or pct >= 100 for pct in pcts):
        raise ValueError("discount_pct must be in 1..99")
    if min(qtys) <= 0:
        raise ValueError("qty must be positive")

    if not seed.currency or not str(seed.currency).strip():
        raise ValueError("currency cannot be empty")

    for up in unit_prices:
        for pct in pcts:
            net = 100 - pct
            # discounted unit price must be integer
            if (up * net) % 100 != 0:
                raise ValueError(
                    "discounted_unit_total_cost requires integer discounted unit price: "
                    "unit_price*(100-discount_pct) must be divisible by 100"
                )

def _validate_vat_total_cost(seed: VatTotalCostSeed) -> None:
    _validate_common_name(seed.name)

    items = as_list_str(seed.item)
    if not items or any(not it.strip() for it in items):
        raise ValueError("item list cannot be empty and items cannot be blank")

    unit_prices = as_list_int(seed.unit_price)
    pcts = as_list_int(seed.vat_pct)
    qtys = as_list_int(seed.qty)

    if not unit_prices or not pcts or not qtys:
        raise ValueError("unit_price/vat_pct/qty lists cannot be empty")

    if min(unit_prices) <= 0:
        raise ValueError("unit_price must be positive")
    if any(pct <= 0 for pct in pcts):
        raise ValueError("vat_pct must be positive")
    if min(qtys) <= 0:
        raise ValueError("qty must be positive")

    if not seed.currency or not str(seed.currency).strip():
        raise ValueError("currency cannot be empty")

    # Ensure integer VAT-applied unit price for EVERY combination
    for up in unit_prices:
        for pct in pcts:
            gross = 100 + pct
            if (up * gross) % 100 != 0:
                raise ValueError(
                    "vat_total_cost requires integer VAT-applied unit price: "
                    "unit_price*(100+vat_pct) must be divisible by 100"
                )

def _validate_bundle_discount_total_cost(seed: BundleDiscountTotalCostSeed) -> None:
    _validate_common_name(seed.name)

    items = as_list_str(seed.item)
    if not items or any(not it.strip() for it in items):
        raise ValueError("item list cannot be empty and items cannot be blank")

    unit_prices = as_list_int(seed.unit_price)
    qtys = as_list_int(seed.qty)
    pcts = as_list_int(seed.discount_pct)

    if not unit_prices or not qtys or not pcts:
        raise ValueError("unit_price/qty/discount_pct lists cannot be empty")

    if min(unit_prices) <= 0:
        raise ValueError("unit_price must be positive")
    if min(qtys) <= 0:
        raise ValueError("qty must be positive")
    if any(pct <= 0 or pct >= 100 for pct in pcts):
        raise ValueError("discount_pct must be in 1..99")

    if not seed.currency or not str(seed.currency).strip():
        raise ValueError("currency cannot be empty")

    # Ensure integer final total for EVERY combination
    for up in unit_prices:
        for q in qtys:
            total = up * q
            for pct in pcts:
                net = 100 - pct
                if (total * net) % 100 != 0:
                    raise ValueError(
                        "bundle_discount_total_cost requires integer final total: "
                        "total_cost*(100-discount_pct) must be divisible by 100 for all combinations"
                    )



