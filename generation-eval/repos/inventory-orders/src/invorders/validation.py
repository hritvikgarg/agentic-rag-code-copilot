"""Input validation helpers. Each returns the (normalised) value or raises ``ValidationError``."""

from __future__ import annotations

import re

from invorders.errors import ValidationError

SKU_PATTERN = re.compile(r"^[A-Z]{3}-\d{4}$")
_POSTCODE = re.compile(r"^\d{5}$")
_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[A-Za-z]{2,}$")
MAX_PRICE_CENTS = 100_000_000


def validate_sku(sku: str) -> str:
    """A SKU is three upper-case letters, a dash and four digits, for example ``ABC-0001``.

    Lower-case input is rejected rather than silently upper-cased.
    """
    if not isinstance(sku, str) or not SKU_PATTERN.match(sku):
        raise ValidationError(
            f"invalid SKU {sku!r}: expected three capital letters, '-', four digits"
        )
    return sku


def validate_quantity(quantity: int, *, maximum: int) -> int:
    """A quantity is an integer from 1 to ``maximum``. Booleans are rejected."""
    if isinstance(quantity, bool) or not isinstance(quantity, int):
        raise ValidationError(f"quantity must be an integer, got {type(quantity).__name__}")
    if quantity < 1:
        raise ValidationError(f"quantity must be at least 1, got {quantity}")
    if quantity > maximum:
        raise ValidationError(f"quantity {quantity} exceeds the maximum of {maximum}")
    return quantity


def validate_price_cents(price: int) -> int:
    """A price is a non-negative integer number of cents, at most ``MAX_PRICE_CENTS``."""
    if isinstance(price, bool) or not isinstance(price, int):
        raise ValidationError(
            f"price must be an integer number of cents, got {type(price).__name__}"
        )
    if price < 0 or price > MAX_PRICE_CENTS:
        raise ValidationError(f"price must be between 0 and {MAX_PRICE_CENTS} cents, got {price}")
    return price


def validate_postcode(postcode: str) -> str:
    """Five digits. Surrounding whitespace is stripped; no other formats are supported."""
    cleaned = postcode.strip() if isinstance(postcode, str) else ""
    if not _POSTCODE.match(cleaned):
        raise ValidationError(f"invalid postcode {postcode!r}: expected exactly five digits")
    return cleaned


def validate_email(email: str) -> str:
    """A deliberately simple e-mail check; the address is lower-cased and stripped."""
    cleaned = email.strip().lower() if isinstance(email, str) else ""
    if not _EMAIL.match(cleaned):
        raise ValidationError(f"invalid e-mail address {email!r}")
    return cleaned
