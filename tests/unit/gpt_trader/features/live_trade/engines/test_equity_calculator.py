from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from gpt_trader.core import Balance
from gpt_trader.features.live_trade.degradation import DegradationState
from gpt_trader.features.live_trade.engines import equity_calculator
from gpt_trader.features.live_trade.engines.equity_calculator import EquityCalculator
from gpt_trader.utilities.async_tools import BoundedToThread


@pytest.mark.asyncio
async def test_equity_calculator_skips_missing_pairs_using_product_list() -> None:
    config = SimpleNamespace(coinbase_default_quote="USDC", read_only=False)
    degradation = DegradationState()
    calculator = EquityCalculator(config, degradation, None, price_history={})

    broker = SimpleNamespace()
    broker.list_balances = MagicMock(
        return_value=[
            Balance(asset="ETH", total=Decimal("1"), available=Decimal("1")),
        ]
    )
    broker.list_products = MagicMock(return_value=[SimpleNamespace(symbol="ETH-USD")])
    broker.get_ticker = MagicMock(return_value={"price": "2000"})

    equity = await calculator.calculate_total_equity(broker, positions={})

    assert equity == Decimal("2000")
    called_products = [call.args[0] for call in broker.get_ticker.call_args_list]
    assert "ETH-USDC" not in called_products
    assert "ETH-USD" in called_products


@pytest.mark.asyncio
async def test_equity_calculator_handles_dict_product_list() -> None:
    config = SimpleNamespace(coinbase_default_quote="USDC", read_only=False)
    degradation = DegradationState()
    calculator = EquityCalculator(config, degradation, None, price_history={})

    broker = SimpleNamespace()
    broker.list_balances = MagicMock(
        return_value=[
            Balance(asset="ERN", total=Decimal("1"), available=Decimal("1")),
        ]
    )
    broker.list_products = MagicMock(return_value=[{"product_id": "ERN-USD"}])
    broker.get_ticker = MagicMock(return_value={"price": "1"})

    equity = await calculator.calculate_total_equity(broker, positions={})

    assert equity == Decimal("1")
    called_products = [call.args[0] for call in broker.get_ticker.call_args_list]
    assert "ERN-USDC" not in called_products
    assert "ERN-USD" in called_products


@pytest.mark.asyncio
async def test_equity_calculator_uses_broker_calls_executor() -> None:
    class RecordingExecutor(ThreadPoolExecutor):
        def __init__(self) -> None:
            super().__init__(max_workers=1)
            self.submit_calls = 0

        def submit(self, *args, **kwargs):
            self.submit_calls += 1
            return super().submit(*args, **kwargs)

    config = SimpleNamespace(coinbase_default_quote="USD", read_only=False)
    degradation = DegradationState()
    executor = RecordingExecutor()
    broker_calls = BoundedToThread(max_concurrency=1, executor=executor)
    calculator = EquityCalculator(
        config,
        degradation,
        None,
        price_history={},
        broker_calls=broker_calls,
    )

    broker = SimpleNamespace()
    broker.list_balances = MagicMock(
        return_value=[
            Balance(asset="USD", total=Decimal("1"), available=Decimal("1")),
        ]
    )
    broker.list_products = MagicMock(return_value=[])
    broker.get_ticker = MagicMock(return_value={"price": "1"})

    equity = await calculator.calculate_total_equity(broker, positions={})

    assert equity == Decimal("1")
    assert executor.submit_calls >= 1
    broker_calls.shutdown()
    executor.shutdown(wait=True)


def _make_diagnostics() -> dict[str, list[str]]:
    return {
        "usd_usdc_found": [],
        "other_assets_found": [],
        "priced_assets": [],
        "unpriced_assets": [],
    }


def test_build_valuation_quotes_prioritizes_default_quote_and_stable_fallbacks() -> None:
    calculator = EquityCalculator(
        SimpleNamespace(coinbase_default_quote="USD", read_only=False),
        DegradationState(),
        None,
        price_history={},
    )

    assert calculator._build_valuation_quotes("EUR") == ["EUR", "USD", "USDC"]
    assert calculator._build_valuation_quotes("USDC") == ["USDC", "USD"]


@pytest.mark.asyncio
async def test_value_asset_skips_missing_known_pair_and_uses_fallback_quote() -> None:
    config = SimpleNamespace(coinbase_default_quote="USDC", read_only=False)
    calculator = EquityCalculator(config, DegradationState(), None, price_history={})

    broker = SimpleNamespace()
    broker.get_ticker = MagicMock(return_value={"price": "2.5"})

    diagnostics = _make_diagnostics()
    valuation_quotes = ["USDC", "USD"]
    known_products = {"CHAIN-USD"}

    usd_value = await calculator._value_asset(
        broker,
        "CHAIN",
        Decimal("2"),
        valuation_quotes,
        diagnostics,
        known_products,
    )

    assert usd_value == Decimal("5.0")
    assert broker.get_ticker.call_args_list == [(("CHAIN-USD",),)]
    assert diagnostics["priced_assets"] == ["CHAIN=2 @ CHAIN-USD≈5.00"]
    assert diagnostics["unpriced_assets"] == []


@pytest.mark.asyncio
async def test_value_asset_records_unpriced_assets_when_ticker_missing_price() -> None:
    config = SimpleNamespace(coinbase_default_quote="USD", read_only=False)
    calculator = EquityCalculator(config, DegradationState(), None, price_history={})

    broker = SimpleNamespace()
    broker.get_ticker = MagicMock(return_value={})

    diagnostics = _make_diagnostics()
    valuation_quotes = ["USD"]
    known_products = {"UNPRICED-USD"}

    usd_value = await calculator._value_asset(
        broker,
        "UNPRICED",
        Decimal("1"),
        valuation_quotes,
        diagnostics,
        known_products,
    )

    assert usd_value is None
    assert diagnostics["priced_assets"] == []
    assert diagnostics["unpriced_assets"] == ["UNPRICED"]


@pytest.mark.asyncio
async def test_get_known_products_returns_cached_set_before_ttl(monkeypatch) -> None:
    config = SimpleNamespace(
        coinbase_default_quote="USD",
        read_only=False,
        product_catalog_ttl_seconds=30,
    )
    calculator = EquityCalculator(config, DegradationState(), None, price_history={})

    calculator._known_products = {"CACHED-USD"}
    calculator._known_products_last_refresh = 1000.0

    time_holder = {"value": 1005.0}

    def fake_time() -> float:
        return time_holder["value"]

    monkeypatch.setattr(equity_calculator.time, "time", fake_time)

    broker = SimpleNamespace()
    result = await calculator._get_known_products(broker)

    assert result == {"CACHED-USD"}
    assert calculator._known_products_last_refresh == 1000.0


@pytest.mark.asyncio
async def test_get_known_products_refreshes_after_ttl(monkeypatch) -> None:
    config = SimpleNamespace(
        coinbase_default_quote="USD",
        read_only=False,
        product_catalog_ttl_seconds=1,
    )
    calculator = EquityCalculator(config, DegradationState(), None, price_history={})

    calculator._known_products = {"OLD-USD"}
    calculator._known_products_last_refresh = 1000.0

    time_holder = {"value": 1005.0}

    def fake_time() -> float:
        return time_holder["value"]

    monkeypatch.setattr(equity_calculator.time, "time", fake_time)

    catalog = SimpleNamespace(_cache={"FRESH-USD": {}})
    broker = SimpleNamespace(product_catalog=catalog)

    result = await calculator._get_known_products(broker)

    assert result == {"FRESH-USD"}
    assert calculator._known_products_last_refresh == 1005.0
