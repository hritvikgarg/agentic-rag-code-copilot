import pytest

from invorders.errors import InsufficientStockError, NotFoundError, ValidationError


def test_receive_unknown_product(world):
    with pytest.raises(NotFoundError):
        world.inventory.receive_stock("ZZZ-9999", 1)


def test_reserve_reduces_available_not_on_hand(world):
    world.inventory.reserve("ORD-1", "CCC-0003", 2)
    level = world.store.get_stock("CCC-0003")
    assert (level.on_hand, level.reserved, level.available) == (3, 2, 1)


def test_reserve_more_than_available_fails(world):
    with pytest.raises(InsufficientStockError) as info:
        world.inventory.reserve("ORD-1", "CCC-0003", 4)
    assert info.value.available == 3


def test_adjust_cannot_go_below_reserved(world):
    world.inventory.reserve("ORD-1", "CCC-0003", 2)
    with pytest.raises(ValidationError):
        world.inventory.adjust_stock("CCC-0003", 1, "stock-take")
    with pytest.raises(ValidationError):
        world.inventory.adjust_stock("CCC-0003", 5, "  ")


def test_release_and_commit(world):
    world.inventory.reserve("ORD-1", "AAA-0001", 10)
    assert world.inventory.release_order("ORD-1") == 1
    assert world.inventory.available("AAA-0001") == 100
    world.inventory.reserve("ORD-2", "AAA-0001", 10)
    assert world.inventory.commit_order("ORD-2") == 1
    level = world.store.get_stock("AAA-0001")
    assert (level.on_hand, level.reserved) == (90, 0)


def test_commit_without_reservations(world):
    with pytest.raises(NotFoundError):
        world.inventory.commit_order("ORD-404")


def test_release_expired_uses_ttl(world):
    world.inventory.reserve("ORD-1", "AAA-0001", 5)
    world.clock.now += world.settings.reservation_ttl_seconds - 1
    assert world.inventory.release_expired() == 0
    world.clock.now += 1
    assert world.inventory.release_expired() == 1
    assert world.inventory.available("AAA-0001") == 100


def test_low_stock_is_sorted(world):
    assert world.inventory.low_stock_skus() == ["CCC-0003"]
