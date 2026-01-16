from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from gpt_trader.core import Balance
from gpt_trader.features.live_trade.degradation import DegradationState
from gpt_trader.features.live_trade.engines.equity_calculator import EquityCalculator


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
