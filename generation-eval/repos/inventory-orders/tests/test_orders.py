import pytest

from invorders.errors import InsufficientStockError, InvalidTransitionError, ValidationError
from invorders.models.order import OrderStatus


def make_order(world, *items):
    order = world.orders.create_order("Ann@Example.com", "12345")
    for sku, qty in items:
        world.orders.add_line(order.order_id, sku, qty)
    return order.order_id


def test_create_normalises_email(world):
    order = world.orders.create_order("Ann@Example.com", "12345")
    assert order.customer_email == "ann@example.com"
    assert order.status is OrderStatus.DRAFT


def test_add_line_merges_same_sku(world):
    oid = make_order(world, ("AAA-0001", 2), ("AAA-0001", 3))
    assert world.store.get_order(oid).lines[0].quantity == 5


def test_price_change_does_not_affect_existing_line(world):
    oid = make_order(world, ("AAA-0001", 1))
    world.store.put_product(world.store.get_product("AAA-0001").with_price(5_000))
    assert world.store.get_order(oid).lines[0].unit_price_cents == 1_000


def test_inactive_product_cannot_be_added(world):
    world.store.put_product(world.store.get_product("BBB-0002").deactivated())
    order = world.orders.create_order("a@b.co", "12345")
    with pytest.raises(ValidationError):
        world.orders.add_line(order.order_id, "BBB-0002", 1)


def test_place_empty_order_fails(world):
    order = world.orders.create_order("a@b.co", "12345")
    with pytest.raises(ValidationError):
        world.orders.place_order(order.order_id)


def test_place_reserves_stock(world):
    oid = make_order(world, ("AAA-0001", 4))
    assert world.orders.place_order(oid).status is OrderStatus.PLACED
    assert world.inventory.available("AAA-0001") == 96


def test_place_rolls_back_partial_reservations(world):
    oid = make_order(world, ("AAA-0001", 4), ("CCC-0003", 3))
    world.inventory.reserve("OTHER", "CCC-0003", 1)  # now only 2 available
    with pytest.raises(InsufficientStockError):
        world.orders.place_order(oid)
    assert world.inventory.available("AAA-0001") == 100
    assert world.store.get_order(oid).status is OrderStatus.DRAFT


def test_lines_are_frozen_after_placing(world):
    oid = make_order(world, ("AAA-0001", 1))
    world.orders.place_order(oid)
    with pytest.raises(ValidationError):
        world.orders.add_line(oid, "BBB-0002", 1)


def test_cancel_placed_releases_stock(world):
    oid = make_order(world, ("AAA-0001", 4))
    world.orders.place_order(oid)
    world.orders.cancel_order(oid)
    assert world.inventory.available("AAA-0001") == 100


def test_ship_commits_stock_and_blocks_cancel(world):
    oid = make_order(world, ("AAA-0001", 4))
    world.orders.place_order(oid)
    world.orders.ship_order(oid)
    assert world.store.get_stock("AAA-0001").on_hand == 96
    with pytest.raises(InvalidTransitionError):
        world.orders.cancel_order(oid)


def test_ship_draft_is_invalid(world):
    oid = make_order(world, ("AAA-0001", 1))
    with pytest.raises(InvalidTransitionError):
        world.orders.ship_order(oid)


def test_events_are_audited(world):
    oid = make_order(world, ("AAA-0001", 1))
    world.orders.place_order(oid)
    topics = [topic for topic, _ in world.audit.entries]
    assert "order.created" in topics and "order.placed" in topics
