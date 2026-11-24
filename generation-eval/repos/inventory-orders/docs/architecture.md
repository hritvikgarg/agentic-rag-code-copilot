# Architecture

`invorders` is organised in three layers.

1. **Models** (`invorders.models`) hold data and local invariants only: a `Product`, a `StockLevel`
   with its `Reservation` records, and an `Order` made of `OrderLine` items with a small status
   machine.
2. **Storage** (`invorders.storage`) defines three repository protocols (`ProductRepository`,
   `StockRepository`, `OrderRepository`) and two implementations. `InMemoryStore` keeps everything
   in dictionaries. `JsonFileStore` implements the same protocols and persists after every write
   to one JSON file, replacing the file atomically.
3. **Services** (`invorders.services`) contain the business rules and depend only on the protocols:
   `InventoryService` (receiving, adjusting, reserving, committing and releasing stock),
   `PricingService` (discounts, tax, shipping) and `OrderService` (the order life cycle).

Cross-cutting pieces: `config.py` (settings), `validation.py` (input checks that raise
`ValidationError`), `events.py` (an event bus plus an audit log) and `errors.py` (one exception
hierarchy rooted at `InvOrdersError`).

## Order life cycle

An order starts as `DRAFT`. Lines can only be added or removed while it is a draft. `place_order`
reserves stock for every line; if any line cannot be reserved, reservations already made for that
order are released and the order stays a draft. A placed order can be cancelled (reservations are
released) or shipped (reservations are committed: `on_hand` and `reserved` both drop by the reserved
quantity). A shipped order can no longer be cancelled.

```
DRAFT --place--> PLACED --ship--> SHIPPED
  |                 |
  +---cancel--------+--cancel--> CANCELLED
```

## Events

Services publish events on the `EventBus` (`order.created`, `order.placed`, `order.cancelled`,
`order.shipped`, `stock.received`, `stock.adjusted`, `stock.reserved`, `stock.released`,
`stock.committed`). Handlers run synchronously in subscription order. An exception in a handler is
caught, logged and recorded in `EventBus.handler_errors`; it never stops the other handlers or
undoes the operation that published the event.

## Pricing

Prices are integer cents. A volume discount applies per line (see `PricingService`). Tax is
computed once on the discounted subtotal, not per line. Shipping is a flat fee that is waived at or
above a configurable subtotal.
