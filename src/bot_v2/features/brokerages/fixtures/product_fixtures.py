"""Loader utilities for brokerage product fixtures."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from decimal import Decimal
from functools import lru_cache
from importlib import resources
from typing import Any

import yaml

from bot_v2.features.brokerages.core.interfaces import MarketType, Product

_FIXTURE_FILENAME = "mock_products.yaml"


@lru_cache(maxsize=1)
def _raw_fixture_payload() -> dict[str, Any]:
    """Return the raw YAML payload bundled with the package."""

    fixture_path = resources.files(__package__).joinpath(_FIXTURE_FILENAME)
    with fixture_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    return payload


def fixture_payload() -> dict[str, Any]:
    """Return a mutable copy of the product fixture payload."""

    return copy.deepcopy(_raw_fixture_payload())


def _section_key(market_type: MarketType) -> str:
    return "perpetual_products" if market_type == MarketType.PERPETUAL else "spot_products"


def _products_section(market_type: MarketType) -> Mapping[str, Any]:
    payload = _raw_fixture_payload()
    return payload.get(_section_key(market_type), {}) or {}


def _parse_market_type(value: Any, *, default: MarketType | None = None) -> MarketType:
    """Normalise fixture values into ``MarketType`` members."""

    if isinstance(value, MarketType):
        return value
    if isinstance(value, str):
        candidate = value.strip()
        if candidate:
            try:
                return MarketType(candidate)
            except ValueError:
                try:
                    return MarketType[candidate.upper()]
                except KeyError as error:
                    raise ValueError(f"Unknown market type {value!r}") from error
    if default is not None:
        return default
    raise ValueError(f"Unknown market type {value!r}")


def list_perpetual_symbols() -> list[str]:
    """Return the known perpetual symbols from fixtures."""

    return list(_products_section(MarketType.PERPETUAL).keys())


def list_spot_symbols() -> list[str]:
    """Return the known spot symbols from fixtures."""

    return list(_products_section(MarketType.SPOT).keys())


def _lookup_product_data(
    symbol: str,
    market_type: MarketType | None,
) -> tuple[dict[str, Any] | None, MarketType]:
    if market_type is not None:
        section = _products_section(market_type)
        entry = section.get(symbol)
        return entry, market_type

    for candidate_type in (MarketType.PERPETUAL, MarketType.SPOT):
        section = _products_section(candidate_type)
        entry = section.get(symbol)
        if entry:
            return entry, candidate_type

    return None, MarketType.PERPETUAL


def create_product(symbol: str, market_type: MarketType | None = None) -> Product:
    """Create a :class:`Product` definition from the bundled fixtures."""

    product_data, resolved_type = _lookup_product_data(symbol, market_type)
    if product_data is None:
        base, _, quote = symbol.partition("-")
        quote = quote or "USD"
        return Product(
            symbol=symbol,
            base_asset=base,
            quote_asset=quote,
            market_type=resolved_type,
            min_size=Decimal("0.001"),
            step_size=Decimal("0.001"),
            min_notional=Decimal("10"),
            price_increment=Decimal("0.01"),
            leverage_max=3,
        )

    return Product(
        symbol=product_data["symbol"],
        base_asset=product_data["base_asset"],
        quote_asset=product_data["quote_asset"],
        market_type=_parse_market_type(product_data.get("market_type"), default=resolved_type),
        min_size=Decimal(product_data["min_size"]),
        step_size=Decimal(product_data["step_size"]),
        min_notional=Decimal(product_data["min_notional"]),
        price_increment=Decimal(product_data["price_increment"]),
        leverage_max=product_data["leverage_max"],
    )


def default_marks() -> dict[str, Decimal]:
    """Return the default mark price mapping from fixtures."""

    payload = _raw_fixture_payload()
    marks = payload.get("default_marks", {}) or {}
    return {symbol: Decimal(str(value)) for symbol, value in marks.items()}


def price_scenario_marks(name: str) -> dict[str, Decimal]:
    """Return mark prices for the named scenario."""

    payload = _raw_fixture_payload()
    scenarios = payload.get("price_scenarios", {}) or {}
    scenario = scenarios.get(name)
    if scenario is None:
        return default_marks()
    return {symbol: Decimal(str(value)) for symbol, value in scenario.items()}


def edge_case_product(symbol: str) -> Product:
    """Return an edge-case product definition from fixtures."""

    payload = _raw_fixture_payload()
    edge_products = payload.get("edge_case_products", {}) or {}
    product_data = edge_products.get(symbol)
    if product_data is None:
        raise ValueError(f"Edge case product {symbol} not found in fixtures")
    return Product(
        symbol=product_data["symbol"],
        base_asset=product_data["base_asset"],
        quote_asset=product_data["quote_asset"],
        market_type=_parse_market_type(
            product_data.get("market_type"), default=MarketType.PERPETUAL
        ),
        min_size=Decimal(product_data["min_size"]),
        step_size=Decimal(product_data["step_size"]),
        min_notional=Decimal(product_data["min_notional"]),
        price_increment=Decimal(product_data["price_increment"]),
        leverage_max=product_data["leverage_max"],
    )
