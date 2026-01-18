"""Contract tests for Coinbase REST PnL service behavior."""

from __future__ import annotations

from decimal import Decimal

from gpt_trader.features.brokerages.coinbase.rest.pnl_service import PnLService
from gpt_trader.features.brokerages.coinbase.utilities import PositionState
from tests.unit.gpt_trader.features.brokerages.coinbase.rest.contract_suite_test_base import (
    CoinbaseRestContractSuiteBase,
)


class TestCoinbaseRestContractSuitePnLService(CoinbaseRestContractSuiteBase):
    def test_process_fill_for_pnl_new_position(self, pnl_service, mock_market_data):
        """Test PnL processing for new position creation."""
        mock_market_data.get_mark.return_value = Decimal("51000")

        fill = {
            "product_id": "BTC-USD",
            "size": "1.0",
            "price": "50000.00",
            "side": "buy",
        }

        pnl_service.process_fill_for_pnl(fill)

        assert "BTC-USD" in pnl_service._position_store.all()
        position = pnl_service._position_store.get("BTC-USD")
        assert position.quantity == Decimal("1.0")
        assert position.entry_price == Decimal("50000.00")

    def test_process_fill_for_pnl_existing_position(self, pnl_service, mock_market_data):
        """Test PnL processing for existing position update."""
        pnl_service._position_store.set(
            "BTC-USD",
            PositionState(
                symbol="BTC-USD",
                side="long",
                quantity=Decimal("1.0"),
                entry_price=Decimal("50000.00"),
            ),
        )
        mock_market_data.get_mark.return_value = Decimal("51000")

        fill = {
            "product_id": "BTC-USD",
            "size": "0.5",
            "price": "51000.00",
            "side": "sell",
        }

        pnl_service.process_fill_for_pnl(fill)

        position = pnl_service._position_store.get("BTC-USD")
        assert position.quantity == Decimal("0.5")
        assert position.realized_pnl == Decimal("500.00")

    def test_process_fill_for_pnl_increases_position(self, pnl_service):
        """Test PnL processing when increasing an existing position."""
        pnl_service._position_store.set(
            "BTC-USD",
            PositionState(
                symbol="BTC-USD",
                side="long",
                quantity=Decimal("1.0"),
                entry_price=Decimal("50000.00"),
            ),
        )

        fill = {
            "product_id": "BTC-USD",
            "size": "0.5",
            "price": "51000.00",
            "side": "buy",
        }

        pnl_service.process_fill_for_pnl(fill)

        position = pnl_service._position_store.get("BTC-USD")
        expected_entry = (Decimal("1.0") * Decimal("50000.00")) + (
            Decimal("0.5") * Decimal("51000.00")
        )
        expected_entry = expected_entry / Decimal("1.5")

        assert position.quantity == Decimal("1.5")
        assert position.entry_price == expected_entry

    def test_process_fill_for_pnl_handles_missing_position_record(self, mock_market_data):
        """Test PnL processing exits when store contains symbol but get returns None."""

        class StoreStub:
            def __init__(self) -> None:
                self.set_calls: list[tuple[str, PositionState]] = []

            def contains(self, symbol: str) -> bool:
                return True

            def get(self, symbol: str) -> PositionState | None:
                return None

            def set(self, symbol: str, position: PositionState) -> None:
                self.set_calls.append((symbol, position))

        store = StoreStub()
        pnl = PnLService(position_store=store, market_data=mock_market_data)

        fill = {
            "product_id": "BTC-USD",
            "size": "0.5",
            "price": "51000.00",
            "side": "sell",
        }

        pnl.process_fill_for_pnl(fill)

        assert store.set_calls == []

    def test_process_fill_for_pnl_short_position_realized_pnl_sign(self, pnl_service):
        """Test realized PnL sign for short positions and zeroed quantity."""
        pnl_service._position_store.set(
            "ETH-USD",
            PositionState(
                symbol="ETH-USD",
                side="short",
                quantity=Decimal("2.0"),
                entry_price=Decimal("100.00"),
                realized_pnl=Decimal("0"),
            ),
        )

        fill = {
            "product_id": "ETH-USD",
            "size": "2.0",
            "price": "90.00",
            "side": "buy",
        }

        pnl_service.process_fill_for_pnl(fill)

        position = pnl_service._position_store.get("ETH-USD")
        assert position.quantity == Decimal("0")
        assert position.realized_pnl == Decimal("20.00")

    def test_process_fill_for_pnl_invalid_data(self, pnl_service):
        """Test PnL processing with invalid fill data."""
        fill = {"product_id": "BTC-USD"}

        pnl_service.process_fill_for_pnl(fill)

        assert not pnl_service._position_store.contains("BTC-USD")

    def test_get_position_pnl_no_position(self, pnl_service):
        """Test position PnL retrieval for non-existent position."""
        pnl = pnl_service.get_position_pnl("BTC-USD")

        assert pnl["symbol"] == "BTC-USD"
        assert pnl["quantity"] == Decimal("0")
        assert pnl["unrealized_pnl"] == Decimal("0")
        assert pnl["realized_pnl"] == Decimal("0")

    def test_get_position_pnl_with_position(self, pnl_service, mock_market_data):
        """Test position PnL retrieval for existing position."""
        pnl_service._position_store.set(
            "BTC-USD",
            PositionState(
                symbol="BTC-USD",
                side="long",
                quantity=Decimal("1.0"),
                entry_price=Decimal("50000.00"),
                realized_pnl=Decimal("1000.00"),
            ),
        )
        mock_market_data.get_mark.return_value = Decimal("51000")

        pnl = pnl_service.get_position_pnl("BTC-USD")

        assert pnl["quantity"] == Decimal("1.0")
        assert pnl["entry"] == Decimal("50000.00")
        assert pnl["mark"] == Decimal("51000")
        assert pnl["unrealized_pnl"] == Decimal("1000.00")
        assert pnl["realized_pnl"] == Decimal("1000.00")

    def test_get_portfolio_pnl_aggregation(self, pnl_service, mock_market_data):
        """Test portfolio PnL aggregation across multiple positions."""
        pnl_service._position_store.set(
            "BTC-USD",
            PositionState(
                symbol="BTC-USD",
                side="long",
                quantity=Decimal("1.0"),
                entry_price=Decimal("50000.00"),
                realized_pnl=Decimal("1000.00"),
            ),
        )
        pnl_service._position_store.set(
            "ETH-USD",
            PositionState(
                symbol="ETH-USD",
                side="short",
                quantity=Decimal("10.0"),
                entry_price=Decimal("3000.00"),
                realized_pnl=Decimal("500.00"),
            ),
        )

        def get_mark_side_effect(symbol):
            if symbol == "BTC-USD":
                return Decimal("51000")
            if symbol == "ETH-USD":
                return Decimal("3000")
            return Decimal("0")

        mock_market_data.get_mark.side_effect = get_mark_side_effect

        portfolio_pnl = pnl_service.get_portfolio_pnl()

        assert portfolio_pnl["total_realized_pnl"] == Decimal("1500.00")
        assert portfolio_pnl["total_unrealized_pnl"] == Decimal("1000.00")
        assert portfolio_pnl["total_pnl"] == Decimal("2500.00")
        assert len(portfolio_pnl["positions"]) == 2
