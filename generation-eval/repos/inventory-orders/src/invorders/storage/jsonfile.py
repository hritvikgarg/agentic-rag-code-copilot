"""A JSON-file store: the in-memory store plus a snapshot written after every change."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from invorders.errors import StorageError
from invorders.models.order import Order, OrderLine, OrderStatus
from invorders.models.product import Product
from invorders.models.stock import Reservation, StockLevel
from invorders.storage.memory import InMemoryStore

FORMAT_VERSION = 1


class JsonFileStore(InMemoryStore):
    """Persists everything to one JSON file, replacing it atomically (write a temporary file in
    the same directory, then ``os.replace``). There is no locking: two processes writing the same
    file can overwrite each other's changes."""

    def __init__(self, path: str | Path) -> None:
        super().__init__()
        self.path = Path(path)
        if self.path.exists():
            self._load()

    # every mutating call flushes
    def put_product(self, product: Product) -> None:
        super().put_product(product)
        self._flush()

    def put_stock(self, level: StockLevel) -> None:
        super().put_stock(level)
        self._flush()

    def add_reservation(self, reservation: Reservation) -> None:
        super().add_reservation(reservation)
        self._flush()

    def remove_reservation(self, reservation_id: str) -> None:
        super().remove_reservation(reservation_id)
        self._flush()

    def put_order(self, order: Order) -> None:
        super().put_order(order)
        self._flush()

    def _flush(self) -> None:
        data = {
            "format_version": FORMAT_VERSION,
            "order_counter": self._order_counter,
            "reservation_counter": self._reservation_counter,
            "products": [vars(p) for p in self.list_products()],
            "stock": [vars(s) for s in self.list_stock()],
            "reservations": [vars(r) for r in self.list_reservations()],
            "orders": [
                {
                    "order_id": o.order_id,
                    "customer_email": o.customer_email,
                    "postcode": o.postcode,
                    "status": o.status.value,
                    "lines": [vars(line) for line in o.lines],
                }
                for o in self.list_orders()
            ],
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(prefix=self.path.name, suffix=".tmp", dir=self.path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(data, handle, indent=2, sort_keys=True)
            os.replace(tmp_name, self.path)
        except OSError as exc:
            Path(tmp_name).unlink(missing_ok=True)
            raise StorageError(f"could not write {self.path.name}: {exc.strerror}") from exc

    def _load(self) -> None:
        try:
            data: dict[str, Any] = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise StorageError(f"cannot read {self.path.name}: {type(exc).__name__}") from exc
        if data.get("format_version") != FORMAT_VERSION:
            raise StorageError(f"unsupported data format version {data.get('format_version')!r}")
        try:
            self._order_counter = int(data["order_counter"])
            self._reservation_counter = int(data["reservation_counter"])
            for p in data["products"]:
                self._products[p["sku"]] = Product(**p)
            for s in data["stock"]:
                self._stock[s["sku"]] = StockLevel(**s)
            for r in data["reservations"]:
                self._reservations[r["reservation_id"]] = Reservation(**r)
            for o in data["orders"]:
                self._orders[o["order_id"]] = Order(
                    order_id=o["order_id"],
                    customer_email=o["customer_email"],
                    postcode=o["postcode"],
                    status=OrderStatus(o["status"]),
                    lines=[OrderLine(**line) for line in o["lines"]],
                )
        except (KeyError, TypeError, ValueError) as exc:
            raise StorageError(f"{self.path.name} is not a valid data file") from exc
