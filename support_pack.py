"""Deterministic support packs for same-skill retry episodes."""

from __future__ import annotations

from typing import Any, Dict


_GENERIC_PACKS: dict[str, dict[str, Any]] = {
    "net_remainder_revenue": {
        "skill_summary": "Önce kalan miktarı bul, sonra birim fiyatla çarp.",
        "formula_hints": ["kalan = toplam - kullanım1 - kullanım2", "gelir = kalan x birim_fiyat"],
        "mini_example": "20 üründen 3 ve 5 tanesi kullanılırsa 12 kalır. Birim fiyat 4 ise gelir 48 olur.",
    },
    "equal_sharing": {
        "skill_summary": "Eşit paylaşımda toplam miktar kişi sayısına bölünür.",
        "formula_hints": ["kişi_başı = toplam / kişi_sayısı"],
        "mini_example": "24 nesne 6 kişiye eşit paylaştırılırsa kişi başı 4 düşer.",
    },
    "percentage_discount": {
        "skill_summary": "İndirimli fiyat, kalan yüzde ile bulunur.",
        "formula_hints": ["kalan_oran = 100 - indirim_yüzdesi", "son_fiyat = eski_fiyat x kalan_oran / 100"],
        "mini_example": "200 TL üründe %10 indirim varsa kalan oran %90, son fiyat 180 TL olur.",
    },
    "percentage_increase": {
        "skill_summary": "Zamlı fiyat, artış sonrası yüzdeyle hesaplanır.",
        "formula_hints": ["yeni_oran = 100 + artış_yüzdesi", "son_fiyat = eski_fiyat x yeni_oran / 100"],
        "mini_example": "200 TL üründe %20 zam varsa yeni oran %120, son fiyat 240 TL olur.",
    },
    "discount_then_vat": {
        "skill_summary": "Sıra önemlidir: önce indirim, sonra KDV eklenir.",
        "formula_hints": [
            "indirimli_birim = eski_birim x (100 - indirim) / 100",
            "kdvli_birim = indirimli_birim x (100 + kdv) / 100",
            "toplam = kdvli_birim x adet",
        ],
        "mini_example": "100 TL ürüne önce %20 indirim uygulanırsa 80 TL olur, sonra %10 KDV ile 88 TL olur.",
    },
}


def build_support_pack(skill_id: str) -> Dict[str, Any]:
    """Return the retry hint pack for a skill."""
    if skill_id in _GENERIC_PACKS:
        pack = _GENERIC_PACKS[skill_id]
    else:
        pack = {
            "skill_summary": "İşlemleri sırayla uygula ve ara sonuçları kontrol et.",
            "formula_hints": [
                "Ara sonucu bulmadan son adıma geçme.",
                "Birden fazla işlem varsa her ara sonucu ayrı yaz.",
            ],
            "mini_example": "Önce ara toplamı bul, sonra son işlemi uygula ve nihai sonucu kontrol et.",
        }
    return {
        "skill_summary": pack["skill_summary"],
        "formula_hints": list(pack["formula_hints"]),
        "mini_example": pack["mini_example"],
    }
