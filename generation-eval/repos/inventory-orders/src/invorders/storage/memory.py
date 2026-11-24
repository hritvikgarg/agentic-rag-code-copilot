"""An in-memory store implementing all three repository protocols (used by tests and demos)."""

from __future__ import annotations

import copy

from invorders.errors import NotFoundError
from invorders.models.order import Order
from invorders.models.product import Product
from invorders.models.stock import Reservation, StockLevel


class InMemoryStore:
    """Dictionaries in one process. Orders are copied on the way in and out so callers cannot
    change stored state without calling ``put_order``."""

    def __init__(self) -> None:
        self._products: dict[str, Product] = {}
        self._stock: dict[str, StockLevel] = {}
        self._reservations: dict[str, Reservation] = {}
        self._orders: dict[str, Order] = {}
        self._order_counter = 0
        self._reservation_counter = 0

    # products
    def get_product(self, sku: str) -> Product:
        try:
            return self._products[sku]
        except KeyError:
            raise NotFoundError(f"no product with SKU {sku}") from None

    def put_product(self, product: Product) -> None:
        self._products[product.sku] = product

    def list_products(self) -> list[Product]:
        return [self._products[k] for k in sorted(self._products)]

    # stock and reservations
    def get_stock(self, sku: str) -> StockLevel:
        return self._stock.get(sku, StockLevel(sku=sku))

    def put_stock(self, level: StockLevel) -> None:
        self._stock[level.sku] = level

    def list_stock(self) -> list[StockLevel]:
        return [self._stock[k] for k in sorted(self._stock)]

    def add_reservation(self, reservation: Reservation) -> None:
        self._reservations[reservation.reservation_id] = reservation

    def reservations_for_order(self, order_id: str) -> list[Reservation]:
        return [r for r in self._reservations.values() if r.order_id == order_id]

    def remove_reservation(self, reservation_id: str) -> None:
        self._reservations.pop(reservation_id, None)

    def list_reservations(self) -> list[Reservation]:
        return list(self._reservations.values())

    def next_reservation_id(self) -> str:
        self._reservation_counter += 1
        return f"RSV-{self._reservation_counter:06d}"

    # orders
    def get_order(self, order_id: str) -> Order:
        try:
            return copy.deepcopy(self._orders[order_id])
        except KeyError:
            raise NotFoundError(f"no order with id {order_id}") from None

    def put_order(self, order: Order) -> None:
        self._orders[order.order_id] = copy.deepcopy(order)

    def list_orders(self) -> list[Order]:
        return [copy.deepcopy(self._orders[k]) for k in sorted(self._orders)]

    def next_order_id(self) -> str:
        self._order_counter += 1
        return f"ORD-{self._order_counter:06d}"
