"""Stock handling: receiving, adjusting, reserving, committing and releasing."""

from __future__ import annotations

import time
from collections.abc import Callable

from invorders.config import Settings
from invorders.errors import InsufficientStockError, NotFoundError, ValidationError
from invorders.events import EventBus
from invorders.models.stock import Reservation, StockLevel
from invorders.storage.base import ProductRepository, StockRepository
from invorders.validation import validate_quantity, validate_sku


class InventoryService:
    """Keeps ``on_hand`` and ``reserved`` consistent. A reservation holds stock for one order."""

    def __init__(
        self,
        stock: StockRepository,
        products: ProductRepository,
        settings: Settings,
        bus: EventBus,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._stock = stock
        self._products = products
        self._settings = settings
        self._bus = bus
        self._clock = clock

    def available(self, sku: str) -> int:
        return self._stock.get_stock(sku).available

    def receive_stock(self, sku: str, quantity: int) -> StockLevel:
        """Add delivered units. The product must already exist in the catalogue."""
        validate_sku(sku)
        validate_quantity(quantity, maximum=self._settings.max_line_quantity)
        self._products.get_product(sku)  # raises NotFoundError for unknown SKUs
        level = self._stock.get_stock(sku)
        updated = StockLevel(sku, level.on_hand + quantity, level.reserved)
        self._stock.put_stock(updated)
        self._bus.publish("stock.received", {"sku": sku, "quantity": quantity})
        return updated

    def adjust_stock(self, sku: str, new_on_hand: int, reason: str) -> StockLevel:
        """Set the physical count (for example after a stock-take). It cannot go below the
        reserved quantity; a reason is mandatory."""
        validate_sku(sku)
        if not reason.strip():
            raise ValidationError("an adjustment needs a reason")
        level = self._stock.get_stock(sku)
        if new_on_hand < level.reserved:
            raise ValidationError(
                f"cannot set on-hand to {new_on_hand}: {level.reserved} units are reserved"
            )
        updated = StockLevel(sku, new_on_hand, level.reserved)
        self._stock.put_stock(updated)
        self._bus.publish("stock.adjusted", {"sku": sku, "on_hand": new_on_hand, "reason": reason})
        return updated

    def reserve(self, order_id: str, sku: str, quantity: int) -> Reservation:
        """Hold ``quantity`` units for ``order_id`` for ``reservation_ttl_seconds``.

        Raises ``InsufficientStockError`` when fewer than ``quantity`` units are available.
        """
        level = self._stock.get_stock(sku)
        if quantity > level.available:
            raise InsufficientStockError(sku, quantity, level.available)
        reservation = Reservation(
            reservation_id=self._stock.next_reservation_id(),
            order_id=order_id,
            sku=sku,
            quantity=quantity,
            expires_at=self._clock() + self._settings.reservation_ttl_seconds,
        )
        self._stock.add_reservation(reservation)
        self._stock.put_stock(StockLevel(sku, level.on_hand, level.reserved + quantity))
        self._bus.publish(
            "stock.reserved", {"order_id": order_id, "sku": sku, "quantity": quantity}
        )
        return reservation

    def release_order(self, order_id: str) -> int:
        """Drop every reservation of an order and return the units to the available pool.
        Returns the number of reservations released (0 if there were none)."""
        released = self._stock.reservations_for_order(order_id)
        for reservation in released:
            level = self._stock.get_stock(reservation.sku)
            self._stock.put_stock(
                StockLevel(level.sku, level.on_hand, level.reserved - reservation.quantity)
            )
            self._stock.remove_reservation(reservation.reservation_id)
        if released:
            self._bus.publish("stock.released", {"order_id": order_id, "count": len(released)})
        return len(released)

    def commit_order(self, order_id: str) -> int:
        """Turn the order's reservations into a physical stock decrease (used when shipping).

        Raises ``NotFoundError`` if the order has no reservations, for example because they
        expired and were released.
        """
        reservations = self._stock.reservations_for_order(order_id)
        if not reservations:
            raise NotFoundError(f"order {order_id} has no reservations to commit")
        for reservation in reservations:
            level = self._stock.get_stock(reservation.sku)
            self._stock.put_stock(
                StockLevel(
                    level.sku,
                    level.on_hand - reservation.quantity,
                    level.reserved - reservation.quantity,
                )
            )
            self._stock.remove_reservation(reservation.reservation_id)
        self._bus.publish("stock.committed", {"order_id": order_id, "count": len(reservations)})
        return len(reservations)

    def release_expired(self, now: float | None = None) -> int:
        """Release reservations whose ``expires_at`` is at or before ``now`` (default: the clock).

        Nothing calls this automatically: expiry happens only when a caller invokes it. The order
        that owned an expired reservation is **not** changed and stays ``PLACED``.
        """
        moment = self._clock() if now is None else now
        expired = [r for r in self._stock.list_reservations() if r.expires_at <= moment]
        for reservation in expired:
            level = self._stock.get_stock(reservation.sku)
            self._stock.put_stock(
                StockLevel(level.sku, level.on_hand, level.reserved - reservation.quantity)
            )
            self._stock.remove_reservation(reservation.reservation_id)
        return len(expired)

    def low_stock_skus(self) -> list[str]:
        """SKUs whose available quantity is at or below ``low_stock_threshold``, sorted."""
        threshold = self._settings.low_stock_threshold
        return sorted(
            p.sku
            for p in self._products.list_products()
            if p.active and self._stock.get_stock(p.sku).available <= threshold
        )
