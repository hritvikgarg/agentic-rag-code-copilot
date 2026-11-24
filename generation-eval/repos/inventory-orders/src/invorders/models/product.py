"""The catalogue entry for one sellable item."""

from __future__ import annotations

from dataclasses import dataclass, replace

from invorders.validation import validate_price_cents, validate_sku


@dataclass(frozen=True)
class Product:
    """A product identified by its SKU. ``unit_price_cents`` is an integer number of cents."""

    sku: str
    name: str
    unit_price_cents: int
    active: bool = True

    def __post_init__(self) -> None:
        validate_sku(self.sku)
        validate_price_cents(self.unit_price_cents)
        if not self.name.strip():
            raise ValueError("a product needs a name")

    def with_price(self, unit_price_cents: int) -> Product:
        """A copy with a new price. Existing order lines keep the price they were created with."""
        return replace(self, unit_price_cents=unit_price_cents)

    def deactivated(self) -> Product:
        """A copy that can no longer be added to orders."""
        return replace(self, active=False)
