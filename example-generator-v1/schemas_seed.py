from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Type, Union, cast

Num = Union[int, List[int]]
StrOrList = Union[str, List[str]]


def as_list_int(x: Num) -> List[int]:
    if isinstance(x, list):
        return [int(v) for v in x]
    return [int(x)]


def as_list_str(x: StrOrList) -> List[str]:
    if isinstance(x, list):
        return [str(v) for v in x]
    return [str(x)]


@dataclass(frozen=True)
class SeedBase:
    template: str
    # name is optional, but recommended. Can be str or list[str]
    name: Optional[StrOrList] = None


# -------------------------
# Tier A templates
# -------------------------

@dataclass(frozen=True)
class ProduceConsumeSellSeed(SeedBase):
    item: str = ""
    produce: Num = 0
    use1: Num = 0
    use2: Num = 0
    price: Num = 0
    currency: str = "TL"
    breakfast_ctx: Optional[str] = None
    dish: Optional[str] = None


@dataclass(frozen=True)
class RemainderAfterLossSeed(SeedBase):
    item: str = ""
    start: Num = 0
    lost: Num = 0


@dataclass(frozen=True)
class EqualSharingSeed(SeedBase):
    item: str = ""
    total: Num = 0
    people: Num = 1


@dataclass(frozen=True)
class MultiStepAddSubSeed(SeedBase):
    item: str = ""
    start: Num = 0
    add: Num = 0
    sub: Num = 0


@dataclass(frozen=True)
class UnitPriceQuantitySeed(SeedBase):
    item: str = ""
    price: Num = 0
    qty: Num = 0
    currency: str = "TL"


@dataclass(frozen=True)
class RateTimeSeed(SeedBase):
    item: str = ""
    rate: Num = 0
    time: Num = 0


# -------------------------
# Phase 2 templates
# -------------------------

@dataclass(frozen=True)
class RatioScalingSeed(SeedBase):
    item_a: str = ""
    item_b: str = ""
    ratio_a: Num = 1
    ratio_b: Num = 1
    total: Num = 0
    ask_side: str = "a"  # "a" or "b"


@dataclass(frozen=True)
class SumAndDifferenceSeed(SeedBase):
    total: Num = 0
    diff: Num = 0


@dataclass(frozen=True)
class CompareDifferenceSeed(SeedBase):
    item: StrOrList = "kalem"
    a: Num = 0
    b: Num = 0


@dataclass(frozen=True)
class ReverseOperationSeed(SeedBase):
    item: StrOrList = "bilye"
    end: Num = 0
    add: Num = 0


# -------------------------
# Phase 3 (Tier B) templates
# -------------------------

# Template 1
@dataclass(frozen=True)
class BuyTwoItemsTotalCostSeed(SeedBase):
    item1: StrOrList = "simit"
    price1: Num = 0
    qty1: Num = 0

    item2: StrOrList = "çay"
    price2: Num = 0
    qty2: Num = 0

    currency: str = "TL"


# Template 2
@dataclass(frozen=True)
class ChangeFromPaymentSeed(SeedBase):
    item1: StrOrList = "kalem"
    price1: Num = 0
    qty1: Num = 0

    item2: StrOrList = "defter"
    price2: Num = 0
    qty2: Num = 0

    paid: Num = 0
    currency: str = "TL"


# Template 3
@dataclass(frozen=True)
class AddThenShareSeed(SeedBase):
    item: StrOrList = "bilye"
    start: Num = 0
    add: Num = 0
    people: Num = 1


# Template 4: Percentage discount -> final price (integer-only)
@dataclass(frozen=True)
class PercentageDiscountFinalPriceSeed(SeedBase):
    item: StrOrList = "mont"
    original_price: Num = 0
    discount_pct: Num = 0  # 10 means %10 indirim
    currency: str = "TL"


# Template 5 (NEW): Percentage increase -> final price (integer-only)
@dataclass(frozen=True)
class PercentageIncreaseFinalPriceSeed(SeedBase):
    item: StrOrList = "mont"
    original_price: Num = 0
    increase_pct: Num = 0  # 10 means %10 zam
    currency: str = "TL"

# Template 6 (NEW): Unit price + % discount + quantity -> total (integer-only)
@dataclass(frozen=True)
class DiscountedUnitTotalCostSeed(SeedBase):
    item: StrOrList = "defter"
    unit_price: Num = 0
    discount_pct: Num = 0  # 10 means %10 indirim
    qty: Num = 0
    currency: str = "TL"

# Template 7(NEW): VAT added to unit price -> total (integer-only)
@dataclass(frozen=True)
class VatTotalCostSeed(SeedBase):
    item: StrOrList = "defter"
    unit_price: Num = 0
    vat_pct: Num = 0  # 10 means %10 KDV
    qty: Num = 0
    currency: str = "TL"

# Template 8 (NEW): Discount applied to total basket -> final total (integer-only)
@dataclass(frozen=True)
class BundleDiscountTotalCostSeed(SeedBase):
    item: StrOrList = "defter"
    unit_price: Num = 0
    qty: Num = 0
    discount_pct: Num = 0  # 10 means %10 indirim
    currency: str = "TL"



# -------------------------
# Tier C templates
# -------------------------

@dataclass(frozen=True)
class DiscountThenVatTotalCostSeed(SeedBase):
    item: StrOrList = "defter"
    unit_price: Num = 0
    discount_pct: Num = 0  # 10 means %10 indirim
    vat_pct: Num = 0       # 20 means %20 KDV
    qty: Num = 0
    currency: str = "TL"


@dataclass(frozen=True)
class TwoItemsThenBundleDiscountTotalCostSeed(SeedBase):
    item1: StrOrList = "defter"
    unit_price1: Num = 0
    qty1: Num = 0

    item2: StrOrList = "kalem"
    unit_price2: Num = 0
    qty2: Num = 0

    discount_pct: Num = 0
    currency: str = "TL"


@dataclass(frozen=True)
class DiscountedTotalThenChangeSeed(SeedBase):
    item: StrOrList = "defter"
    unit_price: Num = 0
    qty: Num = 0
    discount_pct: Num = 0
    paid: Num = 0
    currency: str = "TL"


@dataclass(frozen=True)
class TransformThenShareSeed(SeedBase):
    # Basket discount then equal sharing
    item: StrOrList = "defter"
    unit_price: Num = 0
    qty: Num = 0
    discount_pct: Num = 0
    people: Num = 1
    currency: str = "TL"




Seed = Union[
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
]


_TEMPLATE_MAP: Dict[str, Type[SeedBase]] = {
    # Tier A templates
    "produce_consume_sell": ProduceConsumeSellSeed,
    "sell_leftover": ProduceConsumeSellSeed,  # alias
    "remainder_after_loss": RemainderAfterLossSeed,
    "equal_sharing": EqualSharingSeed,
    "multi_step_add_sub": MultiStepAddSubSeed,
    "unit_price_quantity": UnitPriceQuantitySeed,
    "rate_time": RateTimeSeed,

    "ratio_scaling": RatioScalingSeed,
    "sum_and_difference": SumAndDifferenceSeed,
    "compare_difference": CompareDifferenceSeed,
    "reverse_operation": ReverseOperationSeed,

    # Tier B templates

    "buy_two_items_total_cost": BuyTwoItemsTotalCostSeed,
    "change_from_payment": ChangeFromPaymentSeed,
    "add_then_share": AddThenShareSeed,

    "percentage_discount_final_price": PercentageDiscountFinalPriceSeed,
    "percentage_increase_final_price": PercentageIncreaseFinalPriceSeed,
    "discounted_unit_total_cost": DiscountedUnitTotalCostSeed,
    "vat_total_cost": VatTotalCostSeed,
    "bundle_discount_total_cost": BundleDiscountTotalCostSeed,

    # Tier C templates
    "discount_then_vat_total_cost": DiscountThenVatTotalCostSeed,
    "two_items_then_bundle_discount_total_cost": TwoItemsThenBundleDiscountTotalCostSeed,
    "discounted_total_then_change": DiscountedTotalThenChangeSeed,
    "transform_then_share": TransformThenShareSeed
}


def seed_from_dict(d: Dict[str, Any]) -> Seed:
    if "template" not in d:
        raise ValueError("Seed is missing required field: template")

    template = str(d["template"])
    cls = _TEMPLATE_MAP.get(template)
    if cls is None:
        raise ValueError(f"Unknown template: {template}")

    fields = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    payload: Dict[str, Any] = {}
    for k, v in d.items():
        if k in fields:
            payload[k] = v

    payload["template"] = template
    seed_obj = cls(**payload)  # type: ignore[arg-type]
    return cast(Seed, seed_obj)
