# Configuration

Settings are read by `invorders.config.load_settings` from environment variables that start with
`INVORDERS_`. Any variable with that prefix that is not listed below is rejected with a
`ConfigError`, so typos fail loudly instead of being ignored.

| Variable | Default | Allowed range | Meaning |
|---|---|---|---|
| `INVORDERS_LOW_STOCK_THRESHOLD` | `5` | 0 to 100000 | A SKU is "low" when its available quantity is at or below this |
| `INVORDERS_TAX_RATE_BPS` | `800` | 0 to 5000 | Sales tax in basis points (800 = 8.00 %) |
| `INVORDERS_CURRENCY` | `USD` | three upper-case letters | Currency label used in reports; no conversion is done |
| `INVORDERS_MAX_LINE_QUANTITY` | `1000` | 1 to 100000 | Largest quantity of one SKU on one order |
| `INVORDERS_MAX_ORDER_LINES` | `50` | 1 to 500 | Largest number of distinct SKUs on one order |
| `INVORDERS_RESERVATION_TTL_SECONDS` | `900` | 1 to 86400 | How long a reservation is valid before it can be released as expired |
| `INVORDERS_FREE_SHIPPING_THRESHOLD_CENTS` | `10000` | 0 or more | Subtotal at or above which shipping is free |
| `INVORDERS_SHIPPING_FLAT_CENTS` | `599` | 0 or more | The flat shipping fee |
| `INVORDERS_DATA_DIR` | `./data` | any path | Directory of the JSON data file used by the command-line interface |

An empty value is treated as "not set" and the default is used.
