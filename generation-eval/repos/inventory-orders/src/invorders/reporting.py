"""Human-readable reports."""

from __future__ import annotations

from invorders.config import Settings
from invorders.models.order import Order
from invorders.services.inventory import InventoryService
from invorders.services.pricing import Quote
from invorders.storage.base import ProductRepository


def format_cents(cents: int, currency: str = "USD") -> str:
    """``12345`` -> ``123.45 USD`` (no thousands separator, no currency conversion)."""
    sign = "-" if cents < 0 else ""
    whole, fraction = divmod(abs(cents), 100)
    return f"{sign}{whole}.{fraction:02d} {currency}"


def low_stock_report(
    inventory: InventoryService, products: ProductRepository, settings: Settings
) -> list[str]:
    """One line per low-stock SKU, fewest available first (ties broken by SKU)."""
    rows = []
    for sku in inventory.low_stock_skus():
        rows.append((inventory.available(sku), sku, products.get_product(sku).name))
    rows.sort()
    return [
        f"{sku}  {name}  available={available}  threshold={settings.low_stock_threshold}"
        for available, sku, name in rows
    ]


def order_summary(order: Order, quote: Quote, settings: Settings) -> str:
    """A multi-line summary: header, one line per item, then the totals."""
    lines = [f"Order {order.order_id} [{order.status.value}] for {order.customer_email}"]
    for item in order.lines:
        price = format_cents(item.unit_price_cents, settings.currency)
        lines.append(f"  {item.quantity} x {item.sku} @ {price}")
    lines.append(f"  subtotal {format_cents(quote.subtotal_cents, settings.currency)}")
    lines.append(f"  tax      {format_cents(quote.tax_cents, settings.currency)}")
    lines.append(f"  shipping {format_cents(quote.shipping_cents, settings.currency)}")
    lines.append(f"  total    {format_cents(quote.total_cents, settings.currency)}")
    return "\n".join(lines)
