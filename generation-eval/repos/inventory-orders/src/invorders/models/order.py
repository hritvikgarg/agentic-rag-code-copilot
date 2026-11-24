"""Orders, their lines and the status machine."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from invorders.errors import InvalidTransitionError, ValidationError


class OrderStatus(StrEnum):
    DRAFT = "draft"
    PLACED = "placed"
    SHIPPED = "shipped"
    CANCELLED = "cancelled"


ALLOWED_TRANSITIONS: dict[OrderStatus, frozenset[OrderStatus]] = {
    OrderStatus.DRAFT: frozenset({OrderStatus.PLACED, OrderStatus.CANCELLED}),
    OrderStatus.PLACED: frozenset({OrderStatus.SHIPPED, OrderStatus.CANCELLED}),
    OrderStatus.SHIPPED: frozenset(),
    OrderStatus.CANCELLED: frozenset(),
}


@dataclass(frozen=True)
class OrderLine:
    """One SKU on an order. ``unit_price_cents`` is the price when the line was added."""

    sku: str
    quantity: int
    unit_price_cents: int


@dataclass
class Order:
    """A customer order. Lines can only change while the status is ``DRAFT``."""

    order_id: str
    customer_email: str
    postcode: str
    status: OrderStatus = OrderStatus.DRAFT
    lines: list[OrderLine] = field(default_factory=list)

    def require_draft(self) -> None:
        if self.status is not OrderStatus.DRAFT:
            raise ValidationError(f"order {self.order_id} is {self.status.value}; lines are frozen")

    def transition_to(self, target: OrderStatus) -> None:
        """Move to ``target`` or raise ``InvalidTransitionError``."""
        if target not in ALLOWED_TRANSITIONS[self.status]:
            raise InvalidTransitionError(self.status.value, target.value)
        self.status = target
