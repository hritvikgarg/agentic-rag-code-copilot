"""A minimal command-line interface over the JSON-file store."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from invorders.config import load_settings
from invorders.errors import InvOrdersError
from invorders.events import EventBus
from invorders.reporting import low_stock_report, order_summary
from invorders.services.inventory import InventoryService
from invorders.services.orders import OrderService
from invorders.services.pricing import PricingService
from invorders.storage.jsonfile import JsonFileStore

DATA_FILE_NAME = "inventory.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="invorders", description="Inventory and orders")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("low-stock", help="list SKUs at or below the low-stock threshold")
    show = commands.add_parser("show-order", help="print an order with its price breakdown")
    show.add_argument("order_id")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Exit code 0 on success, 2 when a domain error (``InvOrdersError``) occurs."""
    args = build_parser().parse_args(argv)
    try:
        settings = load_settings()
        store = JsonFileStore(settings.data_dir / DATA_FILE_NAME)
        bus = EventBus()
        inventory = InventoryService(store, store, settings, bus)
        if args.command == "low-stock":
            for line in low_stock_report(inventory, store, settings):
                print(line)
            return 0
        pricing = PricingService(settings)
        orders = OrderService(store, store, inventory, pricing, settings, bus)
        order = store.get_order(args.order_id)
        print(order_summary(order, orders.quote(args.order_id), settings))
        return 0
    except InvOrdersError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
