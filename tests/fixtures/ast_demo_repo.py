"""A small synthetic repository for the structure-aware chunking demos and comparison tests.

``python -m tests.fixtures.ast_demo_repo OUT_DIR`` writes it. It contains a mix on purpose: small
and large functions, decorated and async functions, a class with several methods, a nested class,
a function larger than the default token budget, module-level code, an unparseable Python file,
and non-Python files (Markdown, YAML, JavaScript).
"""

from __future__ import annotations

import sys
from pathlib import Path

SERVICE = '''"""Order service: the synthetic example used by the chunking demos."""
import logging

logger = logging.getLogger(__name__)

TAX_RATE = 0.2


def cache(fn):
    """Trivial decorator."""
    return fn


@cache
def price_with_tax(amount):
    """Return the amount including tax."""
    return amount * (1 + TAX_RATE)


async def fetch_order(order_id):
    """Fetch one order."""
    logger.info("fetching %s", order_id)
    return {"id": order_id}


class OrderRepository:
    """Stores orders."""

    def __init__(self):
        self.orders = {}

    def add(self, order):
        self.orders[order["id"]] = order
        return order

    def get(self, order_id):
        return self.orders.get(order_id)

    class Cursor:
        def next(self):
            return None


if __name__ == "__main__":
    print(price_with_tax(10))
'''


def _big_function() -> str:
    body = "".join(
        f"    stage_{i} = transform(stage_{i - 1} if {i} else data, {i})\n" for i in range(70)
    )
    header = 'def run_pipeline(data):\n    """A function larger than the default token budget."""\n'
    return f"{header}{body}    return stage_69\n"


def _catalog() -> str:
    functions = "\n\n".join(
        f'def item_{i}(x):\n    """Item {i}."""\n    y = x + {i}\n    return y * 2'
        for i in range(24)
    )
    return f'"""Many small functions: longer than one baseline window."""\n\n\n{functions}\n'


def _reports() -> str:
    """Six 22-line functions: with the default 60-line windows (overlap 10) one is cut."""
    functions = "\n\n".join(
        f'def report_{k}(rows):\n    """Report {k}."""\n    total = 0\n'
        + "".join(f"    total += rows[{j}] * {k + 1}\n" for j in range(17))
        + "    return total"
        for k in range(6)
    )
    return f'"""Medium functions."""\n\n{functions}\n'


MODELS = '''"""Plain data classes."""
from dataclasses import dataclass


@dataclass
class Customer:
    name: str
    email: str

    def display(self):
        return f"{self.name} <{self.email}>"


@dataclass
class Item:
    sku: str
    price: float
'''

FILES: dict[str, str] = {
    "shop/service.py": SERVICE,
    "shop/pipeline.py": "import itertools\n\n\n" + _big_function(),
    "shop/models.py": MODELS,
    "shop/catalog.py": _catalog(),
    "shop/reports.py": _reports(),
    "shop/broken.py": "def oops(:\n    pass\n",
    "README.md": "# Shop\n\nA tiny demo.\n\n## Usage\n\nRun the service.\n",
    "config/settings.yaml": "debug: true\nlevel: info\n",
    "web/app.js": "export function render() {\n  return 'ok';\n}\n",
}


def write_demo_repo(root: Path) -> Path:
    for relative, text in FILES.items():
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8", newline="\n")
    return root


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m tests.fixtures.ast_demo_repo OUT_DIR")
    print(write_demo_repo(Path(sys.argv[1])))
