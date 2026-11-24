import json

import pytest

from invorders.errors import NotFoundError, StorageError
from invorders.events import EventBus
from invorders.models.order import Order, OrderLine
from invorders.models.product import Product
from invorders.storage.jsonfile import JsonFileStore
from invorders.storage.memory import InMemoryStore


def test_memory_store_copies_orders():
    store = InMemoryStore()
    store.put_order(Order("ORD-1", "a@b.co", "12345"))
    fetched = store.get_order("ORD-1")
    fetched.lines.append(OrderLine("AAA-0001", 1, 1))
    assert store.get_order("ORD-1").lines == []


def test_missing_records():
    store = InMemoryStore()
    with pytest.raises(NotFoundError):
        store.get_product("AAA-0001")
    assert store.get_stock("AAA-0001").on_hand == 0


def test_json_roundtrip(tmp_path):
    path = tmp_path / "inv.json"
    store = JsonFileStore(path)
    store.put_product(Product("AAA-0001", "Widget", 100))
    store.put_order(Order("ORD-000001", "a@b.co", "12345", lines=[OrderLine("AAA-0001", 2, 100)]))
    reloaded = JsonFileStore(path)
    assert reloaded.get_product("AAA-0001").name == "Widget"
    assert reloaded.get_order("ORD-000001").lines[0].quantity == 2


def test_json_corrupt_file(tmp_path):
    path = tmp_path / "inv.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(StorageError):
        JsonFileStore(path)


def test_json_wrong_version(tmp_path):
    path = tmp_path / "inv.json"
    path.write_text(json.dumps({"format_version": 99}), encoding="utf-8")
    with pytest.raises(StorageError):
        JsonFileStore(path)


def test_failing_subscriber_does_not_break_publisher():
    bus = EventBus()

    def boom(topic, payload):
        raise RuntimeError("x")

    bus.subscribe("t", boom)
    bus.publish("t", {})
    assert bus.handler_errors == [("t", "RuntimeError")]
