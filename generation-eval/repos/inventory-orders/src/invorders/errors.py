"""The exception hierarchy. Every error raised on purpose derives from ``InvOrdersError``."""

from __future__ import annotations


class InvOrdersError(Exception):
    """Base class of all errors raised by this package."""


class ConfigError(InvOrdersError):
    """A setting is missing, malformed, out of range or unknown."""


class ValidationError(InvOrdersError):
    """An input value (SKU, quantity, price, address, ...) is not acceptable."""


class NotFoundError(InvOrdersError):
    """A product, order or reservation does not exist."""


class InsufficientStockError(InvOrdersError):
    """Not enough unreserved stock to satisfy a request."""

    def __init__(self, sku: str, requested: int, available: int) -> None:
        super().__init__(
            f"insufficient stock for {sku}: requested {requested}, available {available}"
        )
        self.sku = sku
        self.requested = requested
        self.available = available


class InvalidTransitionError(InvOrdersError):
    """An order was asked to move to a status it cannot reach from its current status."""

    def __init__(self, current: str, target: str) -> None:
        super().__init__(f"cannot move an order from {current} to {target}")
        self.current = current
        self.target = target


class StorageError(InvOrdersError):
    """The backing store could not be read or written."""
