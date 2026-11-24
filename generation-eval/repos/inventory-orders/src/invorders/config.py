"""Settings loaded from ``INVORDERS_*`` environment variables."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from invorders.errors import ConfigError

ENV_PREFIX = "INVORDERS_"


@dataclass(frozen=True)
class Settings:
    """Validated, immutable settings; ``docs/configuration.md`` explains each field."""

    low_stock_threshold: int = 5
    tax_rate_bps: int = 800
    currency: str = "USD"
    max_line_quantity: int = 1000
    max_order_lines: int = 50
    reservation_ttl_seconds: int = 900
    free_shipping_threshold_cents: int = 10_000
    shipping_flat_cents: int = 599
    data_dir: Path = Path("./data")


# environment variable suffix -> (field name, minimum, maximum)
_INT_FIELDS: dict[str, tuple[str, int, int | None]] = {
    "LOW_STOCK_THRESHOLD": ("low_stock_threshold", 0, 100_000),
    "TAX_RATE_BPS": ("tax_rate_bps", 0, 5_000),
    "MAX_LINE_QUANTITY": ("max_line_quantity", 1, 100_000),
    "MAX_ORDER_LINES": ("max_order_lines", 1, 500),
    "RESERVATION_TTL_SECONDS": ("reservation_ttl_seconds", 1, 86_400),
    "FREE_SHIPPING_THRESHOLD_CENTS": ("free_shipping_threshold_cents", 0, None),
    "SHIPPING_FLAT_CENTS": ("shipping_flat_cents", 0, None),
}
_CURRENCY = re.compile(r"^[A-Z]{3}$")


def _parse_int(name: str, raw: str, low: int, high: int | None) -> int:
    try:
        value = int(raw.strip())
    except ValueError:
        raise ConfigError(f"{ENV_PREFIX}{name} must be an integer, got {raw!r}") from None
    if value < low or (high is not None and value > high):
        bound = f"at least {low}" if high is None else f"between {low} and {high}"
        raise ConfigError(f"{ENV_PREFIX}{name} must be {bound}, got {value}")
    return value


def load_settings(env: Mapping[str, str] | None = None) -> Settings:
    """Build ``Settings`` from ``env`` (default: ``os.environ``).

    Unknown ``INVORDERS_`` variables raise ``ConfigError``. Empty values count as unset.
    """
    source = os.environ if env is None else env
    known = set(_INT_FIELDS) | {"CURRENCY", "DATA_DIR"}
    values: dict[str, object] = {}
    for key, raw in source.items():
        if not key.startswith(ENV_PREFIX):
            continue
        name = key[len(ENV_PREFIX) :]
        if name not in known:
            raise ConfigError(f"unknown setting {key}")
        if raw.strip() == "":
            continue
        if name in _INT_FIELDS:
            field, low, high = _INT_FIELDS[name]
            values[field] = _parse_int(name, raw, low, high)
        elif name == "CURRENCY":
            if not _CURRENCY.match(raw.strip()):
                raise ConfigError(f"{key} must be three upper-case letters, got {raw!r}")
            values["currency"] = raw.strip()
        else:
            values["data_dir"] = Path(raw.strip())
    return Settings(**values)  # type: ignore[arg-type]
