from invorders.reporting import format_cents, low_stock_report, order_summary


def test_format_cents():
    assert format_cents(12345) == "123.45 USD"
    assert format_cents(5) == "0.05 USD"
    assert format_cents(-250, "EUR") == "-2.50 EUR"


def test_low_stock_report_lists_scarce_items(world):
    lines = low_stock_report(world.inventory, world.store, world.settings)
    assert len(lines) == 1 and lines[0].startswith("CCC-0003")


def test_order_summary(world):
    order = world.orders.create_order("a@b.co", "12345")
    world.orders.add_line(order.order_id, "AAA-0001", 2)
    text = order_summary(
        world.store.get_order(order.order_id), world.orders.quote(order.order_id), world.settings
    )
    assert "2 x AAA-0001" in text and "total" in text
