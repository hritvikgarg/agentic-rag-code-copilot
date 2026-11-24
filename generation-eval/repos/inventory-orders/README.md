# invorders

`invorders` is a small inventory and order-management backend written in pure Python (standard
library only). It keeps a catalogue of products, tracks stock per SKU, lets customers build orders,
reserves stock when an order is placed, prices orders (volume discounts, tax, shipping) and ships
them.

It is a **synthetic example project**: it was written from scratch for software-engineering
experiments, contains no real business data and is not meant for production use.

## Layout

| Path | Purpose |
|---|---|
| `src/invorders/config.py` | Settings read from `INVORDERS_*` environment variables |
| `src/invorders/validation.py` | Input validation helpers (SKU, quantity, price, postcode, email) |
| `src/invorders/models/` | Plain data classes: products, stock levels and reservations, orders |
| `src/invorders/storage/` | Repository protocols, an in-memory store and a JSON-file store |
| `src/invorders/services/` | Inventory, pricing and order services (the business logic) |
| `src/invorders/events.py` | A synchronous event bus and an in-memory audit log |
| `src/invorders/reporting.py` | Low-stock report and order summary text |
| `src/invorders/cli.py` | Minimal command-line interface |
| `docs/` | Architecture and configuration notes |
| `tests/` | pytest test suite |

## Quick start

```python
from invorders.config import load_settings
from invorders.events import EventBus
from invorders.models.product import Product
from invorders.services.inventory import InventoryService
from invorders.services.orders import OrderService
from invorders.services.pricing import PricingService
from invorders.storage.memory import InMemoryStore

settings = load_settings({})
store = InMemoryStore()
bus = EventBus()
store.put_product(Product(sku="ABC-0001", name="Widget", unit_price_cents=250))
inventory = InventoryService(store, store, settings, bus)
inventory.receive_stock("ABC-0001", 20)
orders = OrderService(store, store, inventory, PricingService(settings), settings, bus)
order = orders.create_order("buyer@example.com", "12345")
orders.add_line(order.order_id, "ABC-0001", 3)
orders.place_order(order.order_id)
```

See `docs/architecture.md` for how the pieces fit together and `docs/configuration.md` for settings.

## Known limitations

* Single process, single user: stores are not thread-safe and the JSON file store has no file locking.
* Reservations expire only when `InventoryService.release_expired` is called; there is no
  background job.
* Money is handled as integer cents in a single currency; there is no currency conversion.
