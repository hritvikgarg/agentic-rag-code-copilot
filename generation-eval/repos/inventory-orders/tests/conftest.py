"""Shared fixtures: an in-memory world with three products."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from invorders.config import Settings
from invorders.events import AuditLog, EventBus
from invorders.models.product import Product
from invorders.services.inventory import InventoryService
from invorders.services.orders import OrderService
from invorders.services.pricing import PricingService
from invorders.storage.memory import InMemoryStore


class FakeClock:
    def __init__(self, now: float = 1_000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


@dataclass
class World:
    settings: Settings
    store: InMemoryStore
    bus: EventBus
    audit: AuditLog
    clock: FakeClock
    inventory: InventoryService
    pricing: PricingService
    orders: OrderService


@pytest.fixture
def world() -> World:
    settings = Settings()
    store = InMemoryStore()
    bus = EventBus()
    audit = AuditLog(bus)
    clock = FakeClock()
    store.put_product(Product("AAA-0001", "Widget", 1_000))
    store.put_product(Product("BBB-0002", "Gadget", 250))
    store.put_product(Product("CCC-0003", "Gizmo", 9_999))
    inventory = InventoryService(store, store, settings, bus, clock)
    pricing = PricingService(settings)
    orders = OrderService(store, store, inventory, pricing, settings, bus, clock)
    for sku, qty in (("AAA-0001", 100), ("BBB-0002", 20), ("CCC-0003", 3)):
        inventory.receive_stock(sku, qty)
    return World(settings, store, bus, audit, clock, inventory, pricing, orders)
