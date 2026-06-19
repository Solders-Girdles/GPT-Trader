#!/usr/bin/env python3
"""Coinbase product catalog smoke probe."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from gpt_trader.app.config import BotConfig
from gpt_trader.app.container import create_application_container


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="List products visible to the configured broker.")
    parser.add_argument("--limit", type=int, default=50, help="Maximum products to print.")
    return parser.parse_args(argv)


def _field(product: object, *names: str) -> Any:
    if isinstance(product, Mapping):
        for name in names:
            if name in product:
                return product[name]
        return None
    for name in names:
        value = getattr(product, name, None)
        if value is not None:
            return value
    return None


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)

    config = BotConfig.from_env()
    config.dry_run = True
    config.symbols = []

    container = create_application_container(config)

    try:
        products = list(container.broker.list_products())
    except Exception as exc:
        print(f"FAILED to list products: {exc}", file=sys.stderr)
        return 1

    if not products:
        print("No products found. Check API key permissions and network.")
        return 2

    print(f"Found {len(products)} products. Showing first {min(args.limit, len(products))}:")
    for product in products[: args.limit]:
        symbol = _field(product, "product_id", "symbol", "id")
        status = _field(product, "status")
        market_type = _field(product, "product_type", "market_type")
        quote = _field(product, "quote_currency_id", "quote_currency", "quote_asset")
        print(f"- ID: {symbol}, Status: {status}, Type: {market_type}, Quote: {quote}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
