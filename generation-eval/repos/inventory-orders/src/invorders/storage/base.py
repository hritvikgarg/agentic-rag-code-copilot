"""Repository protocols. Services depend on these, never on a concrete store."""

from __future__ import annotations

from typing import Protocol

from invorders.models.order import Order
from invorders.models.product import Product
from invorders.models.stock import Reservation, StockLevel


class ProductRepository(Protocol):
    def get_product(self, sku: str) -> Product:
        """Return the product or raise ``NotFoundError``."""
        ...

    def put_product(self, product: Product) -> None: ...

    def list_products(self) -> list[Product]:
        """All products ordered by SKU."""
        ...


class StockRepository(Protocol):
    def get_stock(self, sku: str) -> StockLevel:
        """The stock level; a SKU never seen before has zero stock (no error)."""
        ...

    def put_stock(self, level: StockLevel) -> None: ...

    def list_stock(self) -> list[StockLevel]: ...

    def add_reservation(self, reservation: Reservation) -> None: ...

    def reservations_for_order(self, order_id: str) -> list[Reservation]: ...

    def remove_reservation(self, reservation_id: str) -> None: ...

    def list_reservations(self) -> list[Reservation]: ...

    def next_reservation_id(self) -> str: ...


class OrderRepository(Protocol):
    def get_order(self, order_id: str) -> Order:
        """Return the order or raise ``NotFoundError``."""
        ...

    def put_order(self, order: Order) -> None: ...

    def list_orders(self) -> list[Order]: ...

    def next_order_id(self) -> str: ...
