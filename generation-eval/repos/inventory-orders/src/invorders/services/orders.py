"""The order life cycle: create, edit lines, place, cancel and ship."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import replace

from invorders.config import Settings
from invorders.errors import InsufficientStockError, ValidationError
from invorders.events import EventBus
from invorders.models.order import Order, OrderLine, OrderStatus
from invorders.services.inventory import InventoryService
from invorders.services.pricing import PricingService, Quote
from invorders.storage.base import OrderRepository, ProductRepository
from invorders.validation import validate_email, validate_postcode, validate_quantity, validate_sku


class OrderService:
    def __init__(
        self,
        orders: OrderRepository,
        products: ProductRepository,
        inventory: InventoryService,
        pricing: PricingService,
        settings: Settings,
        bus: EventBus,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._orders = orders
        self._products = products
        self._inventory = inventory
        self._pricing = pricing
        self._settings = settings
        self._bus = bus
        self._clock = clock

    def create_order(self, customer_email: str, postcode: str) -> Order:
        order = Order(
            order_id=self._orders.next_order_id(),
            customer_email=validate_email(customer_email),
            postcode=validate_postcode(postcode),
        )
        self._orders.put_order(order)
        self._bus.publish("order.created", {"order_id": order.order_id})
        return order

    def add_line(self, order_id: str, sku: str, quantity: int) -> Order:
        """Add ``quantity`` units of ``sku`` to a draft order.

        The unit price is copied from the catalogue now; a later price change does not affect it.
        Adding a SKU that is already on the order increases that line (the combined quantity must
        still respect ``max_line_quantity``).
        """
        order = self._orders.get_order(order_id)
        order.require_draft()
        validate_sku(sku)
        validate_quantity(quantity, maximum=self._settings.max_line_quantity)
        product = self._products.get_product(sku)
        if not product.active:
            raise ValidationError(f"product {sku} is not active and cannot be ordered")
        existing = next((line for line in order.lines if line.sku == sku), None)
        if existing is None:
            if len(order.lines) >= self._settings.max_order_lines:
                raise ValidationError(
                    f"an order can have at most {self._settings.max_order_lines} lines"
                )
            order.lines.append(OrderLine(sku, quantity, product.unit_price_cents))
        else:
            combined = existing.quantity + quantity
            validate_quantity(combined, maximum=self._settings.max_line_quantity)
            order.lines[order.lines.index(existing)] = replace(existing, quantity=combined)
        self._orders.put_order(order)
        return order

    def remove_line(self, order_id: str, sku: str) -> Order:
        order = self._orders.get_order(order_id)
        order.require_draft()
        remaining = [line for line in order.lines if line.sku != sku]
        if len(remaining) == len(order.lines):
            raise ValidationError(f"order {order_id} has no line for {sku}")
        order.lines = remaining
        self._orders.put_order(order)
        return order

    def place_order(self, order_id: str) -> Order:
        """Reserve stock for every line and mark the order placed.

        If any line cannot be reserved, the reservations already made for this order are released,
        the order stays a draft and ``InsufficientStockError`` is re-raised.
        """
        order = self._orders.get_order(order_id)
        order.require_draft()
        if not order.lines:
            raise ValidationError("an order needs at least one line before it can be placed")
        try:
            for line in order.lines:
                self._inventory.reserve(order.order_id, line.sku, line.quantity)
        except InsufficientStockError:
            self._inventory.release_order(order.order_id)
            raise
        order.transition_to(OrderStatus.PLACED)
        self._orders.put_order(order)
        self._bus.publish("order.placed", {"order_id": order.order_id})
        return order

    def cancel_order(self, order_id: str) -> Order:
        """Cancel a draft or placed order (placed orders give their reservations back).
        A shipped order cannot be cancelled."""
        order = self._orders.get_order(order_id)
        was_placed = order.status is OrderStatus.PLACED
        order.transition_to(OrderStatus.CANCELLED)
        if was_placed:
            self._inventory.release_order(order.order_id)
        self._orders.put_order(order)
        self._bus.publish("order.cancelled", {"order_id": order.order_id})
        return order

    def ship_order(self, order_id: str) -> Order:
        """Ship a placed order: commit its reservations, then mark it shipped."""
        order = self._orders.get_order(order_id)
        if order.status is not OrderStatus.PLACED:
            order.transition_to(OrderStatus.SHIPPED)  # raises InvalidTransitionError
        self._inventory.commit_order(order.order_id)
        order.transition_to(OrderStatus.SHIPPED)
        self._orders.put_order(order)
        self._bus.publish("order.shipped", {"order_id": order.order_id})
        return order

    def quote(self, order_id: str) -> Quote:
        return self._pricing.quote(self._orders.get_order(order_id))
