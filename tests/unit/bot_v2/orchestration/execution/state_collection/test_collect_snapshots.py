"""Tests for state collection and account snapshot functionality."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from bot_v2.features.brokerages.core.interfaces import Balance
from bot_v2.orchestration.execution.state_collection import StateCollector


class TestStateCollectorInitialization:
    """Test StateCollector initialization and configuration."""

    def test_initialization_with_default_settings(self, mock_brokerage) -> None:
        """Test StateCollector initialization with default runtime settings."""
        collector = StateCollector(mock_brokerage)

        assert collector.broker == mock_brokerage
        assert collector._settings is not None
        assert "USD" in collector.collateral_assets
        assert "USDC" in collector.collateral_assets

    def test_initialization_with_custom_settings(
        self, mock_brokerage, mock_runtime_settings
    ) -> None:
        """Test StateCollector initialization with custom runtime settings."""
        collector = StateCollector(mock_brokerage, settings=mock_runtime_settings)

        assert collector.broker == mock_brokerage
        assert collector._settings == mock_runtime_settings
        assert "USD" in collector.collateral_assets
        assert "USDC" in collector.collateral_assets
        assert "ETH" in collector.collateral_assets

    def test_collateral_assets_resolution_from_env(self, mock_runtime_settings) -> None:
        """Test collateral assets resolution from environment variable."""
        collector = StateCollector(MagicMock(), settings=mock_runtime_settings)

        expected_assets = {"USD", "USDC", "ETH"}
        assert collector.collateral_assets == expected_assets

    def test_collateral_assets_resolution_defaults(self, mock_brokerage) -> None:
        """Test collateral assets resolution defaults when env is empty."""
        # Mock settings with empty env
        mock_settings = MagicMock()
        mock_settings.raw_env = {"PERPS_COLLATERAL_ASSETS": ""}

        collector = StateCollector(mock_brokerage, settings=mock_settings)

        expected_assets = {"USD", "USDC"}
        assert collector.collateral_assets == expected_assets

    def test_collateral_assets_resolution_whitespace_handling(self, mock_brokerage) -> None:
        """Test collateral assets resolution handles whitespace properly."""
        mock_settings = MagicMock()
        mock_settings.raw_env = {"PERPS_COLLATERAL_ASSETS": " USD , USDC ,  ETH  "}

        collector = StateCollector(mock_brokerage, settings=mock_settings)

        expected_assets = {"USD", "USDC", "ETH"}
        assert collector.collateral_assets == expected_assets

    def test_collateral_assets_case_insensitive(self, mock_brokerage) -> None:
        """Test collateral assets resolution is case insensitive."""
        mock_settings = MagicMock()
        mock_settings.raw_env = {"PERPS_COLLATERAL_ASSETS": "usd,usdc,btc"}

        collector = StateCollector(mock_brokerage, settings=mock_settings)

        expected_assets = {"USD", "USDC", "BTC"}
        assert collector.collateral_assets == expected_assets


class TestCollateralAssetResolution:
    """Test collateral asset resolution functionality."""

    def test_resolve_collateral_assets_empty_env(self, mock_brokerage) -> None:
        """Test _resolve_collateral_assets with empty environment."""
        mock_settings = MagicMock()
        mock_settings.raw_env = {}
        collector = StateCollector(mock_brokerage, settings=mock_settings)

        expected = {"USD", "USDC"}
        assert collector.collateral_assets == expected

    def test_resolve_collateral_assets_malformed_env(self, mock_brokerage) -> None:
        """Test _resolve_collateral_assets with malformed environment."""
        mock_settings = MagicMock()
        mock_settings.raw_env = {"PERPS_COLLATERAL_ASSETS": ",,,,"}
        collector = StateCollector(mock_brokerage, settings=mock_settings)

        expected = {"USD", "USDC"}
        assert collector.collateral_assets == expected

    def test_resolve_collateral_assets_single_asset(self, mock_brokerage) -> None:
        """Test _resolve_collateral_assets with single asset."""
        mock_settings = MagicMock()
        mock_settings.raw_env = {"PERPS_COLLATERAL_ASSETS": "BTC"}
        collector = StateCollector(mock_brokerage, settings=mock_settings)

        expected = {"BTC"}
        assert collector.collateral_assets == expected


class TestCalculateEquityFromBalances:
    """Test equity calculation from balance lists."""

    def test_calculate_equity_with_collateral_assets(
        self, state_collector, sample_balances
    ) -> None:
        """Test equity calculation with multiple collateral assets."""
        equity, collateral_balances, total_balance = state_collector.calculate_equity_from_balances(
            sample_balances
        )

        # Should include USD, USDC (collateral assets)
        expected_equity = Decimal("15000.0") + Decimal("25000.0")  # USD + USDC
        assert equity == expected_equity
        assert len(collateral_balances) == 2
        assert total_balance == expected_equity

    def test_calculate_equity_no_collateral_assets(self, state_collector) -> None:
        """Test equity calculation with no matching collateral assets."""
        non_collateral_balances = [
            Balance(asset="ETH", available=Decimal("2.0"), total=Decimal("2.0"), hold=Decimal("0")),
            Balance(asset="BTC", available=Decimal("0.1"), total=Decimal("0.1"), hold=Decimal("0")),
        ]

        equity, collateral_balances, total_balance = state_collector.calculate_equity_from_balances(
            non_collateral_balances
        )

        assert equity == Decimal("0")
        assert collateral_balances == []
        assert total_balance == Decimal("0")

    def test_calculate_equity_fallback_to_usd(self, state_collector) -> None:
        """Test equity calculation falls back to USD when no collateral assets found."""
        balances_with_usd = [
            Balance(asset="ETH", available=Decimal("2.0"), total=Decimal("2.0"), hold=Decimal("0")),
            Balance(
                asset="USD", available=Decimal("1000.0"), total=Decimal("1000.0"), hold=Decimal("0")
            ),
            Balance(asset="BTC", available=Decimal("0.1"), total=Decimal("0.1"), hold=Decimal("0")),
        ]

        equity, collateral_balances, total_balance = state_collector.calculate_equity_from_balances(
            balances_with_usd
        )

        assert equity == Decimal("1000.0")
        assert len(collateral_balances) == 1
        assert collateral_balances[0].asset == "USD"
        assert total_balance == Decimal("1000.0")

    def test_calculate_equity_no_usd_fallback(self, state_collector) -> None:
        """Test equity calculation when no USD balance available."""
        balances_no_usd = [
            Balance(asset="ETH", available=Decimal("2.0"), total=Decimal("2.0"), hold=Decimal("0")),
            Balance(asset="BTC", available=Decimal("0.1"), total=Decimal("0.1"), hold=Decimal("0")),
        ]

        equity, collateral_balances, total_balance = state_collector.calculate_equity_from_balances(
            balances_no_usd
        )

        assert equity == Decimal("0")
        assert collateral_balances == []
        assert total_balance == Decimal("0")

    def test_calculate_equity_complex_scenarios(self, state_collector, complex_balances) -> None:
        """Test equity calculation with complex balance scenarios."""
        equity, collateral_balances, total_balance = state_collector.calculate_equity_from_balances(
            complex_balances
        )

        # Should include USD (1000.0) and USDC (0.0) and usdc (2000.0) - case insensitive matching
        # Should exclude negative ETH, empty/None assets
        expected_equity = (
            Decimal("1000.0") + Decimal("0.0") + Decimal("2000.0")
        )  # USD + USDC + usdc
        assert equity == expected_equity
        assert len(collateral_balances) == 3  # USD, USDC, and usdc (case insensitive)
        assert total_balance == expected_equity

    def test_calculate_equity_handles_zero_balances(self, state_collector) -> None:
        """Test equity calculation with zero balances."""
        zero_balances = [
            Balance(asset="USD", available=Decimal("0.0"), total=Decimal("0.0"), hold=Decimal("0")),
            Balance(
                asset="USDC", available=Decimal("0.0"), total=Decimal("0.0"), hold=Decimal("0")
            ),
        ]

        equity, collateral_balances, total_balance = state_collector.calculate_equity_from_balances(
            zero_balances
        )

        assert equity == Decimal("0")
        assert len(collateral_balances) == 2
        assert total_balance == Decimal("0")

    def test_calculate_equity_negative_balances(self, state_collector) -> None:
        """Test equity calculation with negative balances."""
        negative_balances = [
            Balance(
                asset="USD", available=Decimal("-100.0"), total=Decimal("-100.0"), hold=Decimal("0")
            ),
            Balance(
                asset="USDC", available=Decimal("500.0"), total=Decimal("500.0"), hold=Decimal("0")
            ),
        ]

        equity, collateral_balances, total_balance = state_collector.calculate_equity_from_balances(
            negative_balances
        )

        # Should include both balances in calculation
        expected_equity = Decimal("-100.0") + Decimal("500.0")
        assert equity == expected_equity
        assert len(collateral_balances) == 2
        assert total_balance == expected_equity


class TestCollectAccountState:
    """Test complete account state collection."""

    def test_collect_account_state_success(
        self, state_collector, sample_balances, sample_positions
    ) -> None:
        """Test successful account state collection."""
        state_collector.broker.list_balances.return_value = sample_balances
        state_collector.broker.list_positions.return_value = sample_positions

        balances, equity, collateral_balances, total_balance, positions = (
            state_collector.collect_account_state()
        )

        # Verify all components are returned
        assert balances == sample_balances
        assert equity > Decimal("0")
        assert collateral_balances is not None
        assert total_balance > Decimal("0")
        assert positions == sample_positions

        # Verify broker methods were called
        state_collector.broker.list_balances.assert_called_once()
        state_collector.broker.list_positions.assert_called_once()

    def test_collect_account_state_empty_balances(self, state_collector) -> None:
        """Test account state collection with empty balances."""
        state_collector.broker.list_balances.return_value = []
        state_collector.broker.list_positions.return_value = []

        balances, equity, collateral_balances, total_balance, positions = (
            state_collector.collect_account_state()
        )

        assert balances == []
        assert equity == Decimal("0")
        assert collateral_balances == []
        assert total_balance == Decimal("0")
        assert positions == []

    def test_collect_account_state_empty_positions(self, state_collector, sample_balances) -> None:
        """Test account state collection with empty positions."""
        state_collector.broker.list_balances.return_value = sample_balances
        state_collector.broker.list_positions.return_value = []

        balances, equity, collateral_balances, total_balance, positions = (
            state_collector.collect_account_state()
        )

        assert balances == sample_balances
        assert equity > Decimal("0")
        assert collateral_balances is not None
        assert total_balance > Decimal("0")
        assert positions == []

    def test_collect_account_state_broker_errors(self, state_collector) -> None:
        """Test account state collection handles broker errors gracefully."""
        state_collector.broker.list_balances.side_effect = RuntimeError(
            "Balance service unavailable"
        )

        with pytest.raises(RuntimeError, match="Balance service unavailable"):
            state_collector.collect_account_state()

    def test_collect_account_state_integration(
        self, state_collector_with_settings, sample_balances
    ) -> None:
        """Test account state collection with custom settings integration."""
        state_collector_with_settings.broker.list_balances.return_value = sample_balances
        state_collector_with_settings.broker.list_positions.return_value = []

        balances, equity, collateral_balances, total_balance, positions = (
            state_collector_with_settings.collect_account_state()
        )

        # Should use custom collateral assets (USD, USDC, ETH)
        expected_equity = (
            Decimal("15000.0") + Decimal("25000.0") + Decimal("2.0")
        )  # USD + USDC + ETH
        assert equity == expected_equity
        assert len(collateral_balances) == 3  # USD, USDC, and ETH
