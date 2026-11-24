"""Pricing: volume discounts, tax and shipping. All amounts are integer cents."""

from __future__ import annotations

from dataclasses import dataclass

from invorders.config import Settings
from invorders.models.order import Order, OrderLine

#: (minimum quantity, discount percent), checked from the largest threshold down.
VOLUME_DISCOUNT_TIERS: tuple[tuple[int, int], ...] = ((50, 10), (10, 5))


@dataclass(frozen=True)
class Quote:
    """The price breakdown of an order."""

    gross_cents: int  # sum of quantity x unit price before discounts
    discount_cents: int  # total volume discount
    subtotal_cents: int  # gross - discount
    tax_cents: int
    shipping_cents: int
    total_cents: int  # subtotal + tax + shipping


class PricingService:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    @staticmethod
    def volume_discount_percent(quantity: int) -> int:
        """5 % from 10 units of one SKU, 10 % from 50 units; the discount is per order line."""
        for minimum, percent in VOLUME_DISCOUNT_TIERS:
            if quantity >= minimum:
                return percent
        return 0

    def line_gross_cents(self, line: OrderLine) -> int:
        return line.quantity * line.unit_price_cents

    def line_discount_cents(self, line: OrderLine) -> int:
        """The discount rounded **down** to a whole cent (``gross * percent // 100``)."""
        return self.line_gross_cents(line) * self.volume_discount_percent(line.quantity) // 100

    def tax_cents(self, subtotal_cents: int) -> int:
        """Tax on the discounted subtotal, rounded half up, computed once for the whole order
        (not per line). Shipping is not taxed."""
        return (subtotal_cents * self._settings.tax_rate_bps + 5_000) // 10_000

    def shipping_cents(self, subtotal_cents: int) -> int:
        """Free for an empty order and at or above the free-shipping threshold; otherwise flat."""
        if subtotal_cents == 0 or subtotal_cents >= self._settings.free_shipping_threshold_cents:
            return 0
        return self._settings.shipping_flat_cents

    def quote(self, order: Order) -> Quote:
        gross = sum(self.line_gross_cents(line) for line in order.lines)
        discount = sum(self.line_discount_cents(line) for line in order.lines)
        subtotal = gross - discount
        tax = self.tax_cents(subtotal)
        shipping = self.shipping_cents(subtotal)
        return Quote(gross, discount, subtotal, tax, shipping, subtotal + tax + shipping)
