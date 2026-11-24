import pytest

from invorders.config import Settings
from invorders.models.order import Order, OrderLine
from invorders.services.pricing import PricingService


@pytest.fixture
def pricing():
    return PricingService(Settings())


@pytest.mark.parametrize("quantity,percent", [(1, 0), (9, 0), (10, 5), (49, 5), (50, 10)])
def test_volume_discount_tiers(quantity, percent):
    assert PricingService.volume_discount_percent(quantity) == percent


def test_discount_rounds_down(pricing):
    line = OrderLine("AAA-0001", 10, 333)  # gross 3330, 5 % = 166.5 -> 166
    assert pricing.line_discount_cents(line) == 166


def test_tax_rounds_half_up(pricing):
    assert pricing.tax_cents(1_000) == 80
    assert pricing.tax_cents(63) == 5  # 5.04 -> 5
    assert pricing.tax_cents(62) == 5  # 4.96 -> 5
    assert pricing.tax_cents(6) == 0  # 0.48 -> 0


def test_shipping_rules(pricing):
    assert pricing.shipping_cents(0) == 0
    assert pricing.shipping_cents(9_999) == 599
    assert pricing.shipping_cents(10_000) == 0


def test_quote_total(pricing):
    order = Order("ORD-000001", "a@b.co", "12345", lines=[OrderLine("AAA-0001", 10, 1_000)])
    quote = pricing.quote(order)
    assert (quote.gross_cents, quote.discount_cents, quote.subtotal_cents) == (10_000, 500, 9_500)
    assert quote.tax_cents == 760
    assert quote.shipping_cents == 599
    assert quote.total_cents == 9_500 + 760 + 599
