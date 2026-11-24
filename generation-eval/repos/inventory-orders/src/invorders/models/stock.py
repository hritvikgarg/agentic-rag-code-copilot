"""Stock levels and reservations."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StockLevel:
    """Physical stock (``on_hand``) and the part of it promised to placed orders (``reserved``)."""

    sku: str
    on_hand: int = 0
    reserved: int = 0

    def __post_init__(self) -> None:
        if self.on_hand < 0 or self.reserved < 0 or self.reserved > self.on_hand:
            raise ValueError("require 0 <= reserved <= on_hand")

    @property
    def available(self) -> int:
        """Units that can still be reserved."""
        return self.on_hand - self.reserved


@dataclass(frozen=True)
class Reservation:
    """A hold on ``quantity`` units of ``sku`` for one order until ``expires_at`` (epoch s)."""

    reservation_id: str
    order_id: str
    sku: str
    quantity: int
    expires_at: float
