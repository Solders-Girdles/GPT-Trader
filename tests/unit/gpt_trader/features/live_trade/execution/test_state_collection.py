"""Tests for `StateCollector` initialization, collateral resolution, and positions dict."""

from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import MagicMock

from gpt_trader.features.live_trade.execution.state_collection import StateCollector


class TestStateCollectorInit:
    """Tests for StateCollector initialization."""

    def test_init_stores_broker(self, mock_broker: MagicMock, mock_config) -> None:
        """Test that broker is stored correctly."""
        collector = StateCollector(mock_broker, mock_config)
        assert collector.broker is mock_broker

    def test_init_resolves_collateral_assets(
        self, mock_broker: MagicMock, mock_config, monkeypatch
    ) -> None:
        """Test that collateral assets are resolved from environment."""
        monkeypatch.setenv("PERPS_COLLATERAL_ASSETS", "USDT,DAI,USDC")
        collector = StateCollector(mock_broker, mock_config)
        assert collector.collateral_assets == {"USDT", "DAI", "USDC"}

    def test_init_uses_defaults_when_no_env(
        self, mock_broker: MagicMock, mock_config, monkeypatch
    ) -> None:
        """Test that default collateral assets are used when env is empty."""
        monkeypatch.delenv("PERPS_COLLATERAL_ASSETS", raising=False)
        collector = StateCollector(mock_broker, mock_config)
        assert collector.collateral_assets == {"USD", "USDC"}

    def test_init_integration_mode_explicit_parameter(
        self, mock_broker: MagicMock, mock_config
    ) -> None:
        """Test integration mode can be set via explicit parameter."""
        collector = StateCollector(mock_broker, mock_config, integration_mode=True)
        assert collector._integration_mode is True

    def test_init_integration_mode_defaults_false(
        self, mock_broker: MagicMock, mock_config
    ) -> None:
        """Test integration_mode defaults to False."""
        collector = StateCollector(mock_broker, mock_config)
        assert collector._integration_mode is False


class TestResolveCollateralAssets:
    """Tests for _resolve_collateral_assets method."""

    def test_parses_comma_separated_values(
        self, mock_broker: MagicMock, mock_config, monkeypatch
    ) -> None:
        """Test parsing comma-separated collateral assets."""
        monkeypatch.setenv("PERPS_COLLATERAL_ASSETS", "USD, USDC, USDT")
        collector = StateCollector(mock_broker, mock_config)
        assert collector.collateral_assets == {"USD", "USDC", "USDT"}

    def test_uppercase_normalization(
        self, mock_broker: MagicMock, mock_config, monkeypatch
    ) -> None:
        """Test that asset names are uppercased."""
        monkeypatch.setenv("PERPS_COLLATERAL_ASSETS", "usd,usdc")
        collector = StateCollector(mock_broker, mock_config)
        assert collector.collateral_assets == {"USD", "USDC"}

    def test_empty_returns_defaults(self, mock_broker: MagicMock, mock_config, monkeypatch) -> None:
        """Test that empty env returns defaults."""
        monkeypatch.setenv("PERPS_COLLATERAL_ASSETS", "")
        collector = StateCollector(mock_broker, mock_config)
        assert collector.collateral_assets == {"USD", "USDC"}


class TestBuildPositionsDict:
    """Tests for build_positions_dict method."""

    def test_builds_dict_from_positions(self, collector: StateCollector) -> None:
        """Test building position dictionary."""
        positions = [
            SimpleNamespace(
                symbol="BTC-PERP",
                quantity=Decimal("1.5"),
                side="long",
                entry_price=Decimal("50000"),
                mark_price=Decimal("51000"),
            ),
        ]

        result = collector.build_positions_dict(positions)

        assert "BTC-PERP" in result
        assert result["BTC-PERP"]["quantity"] == Decimal("1.5")
        assert result["BTC-PERP"]["side"] == "long"
        assert result["BTC-PERP"]["entry_price"] == Decimal("50000")
        assert result["BTC-PERP"]["mark_price"] == Decimal("51000")

    def test_skips_zero_quantity_positions(self, collector: StateCollector) -> None:
        """Test that zero quantity positions are skipped."""
        positions = [
            SimpleNamespace(
                symbol="BTC-PERP",
                quantity=Decimal("0"),
                side="long",
                entry_price=Decimal("50000"),
                mark_price=Decimal("51000"),
            ),
        ]

        result = collector.build_positions_dict(positions)

        assert len(result) == 0

    def test_handles_parse_errors(self, collector: StateCollector) -> None:
        """Test that parse errors are handled gracefully."""
        positions = [
            SimpleNamespace(
                symbol="BTC-PERP",
                quantity=Decimal("1.5"),
                side="long",
                entry_price="invalid",  # Will cause Decimal conversion error
                mark_price=Decimal("51000"),
            ),
        ]

        result = collector.build_positions_dict(positions)

        assert len(result) == 0

    def test_defaults_side_to_long(self, collector: StateCollector) -> None:
        """Test that side defaults to 'long'."""
        positions = [
            SimpleNamespace(
                symbol="BTC-PERP",
                quantity=Decimal("1.5"),
                # No side attribute
                entry_price=Decimal("50000"),
                mark_price=Decimal("51000"),
            ),
        ]

        result = collector.build_positions_dict(positions)

        assert result["BTC-PERP"]["side"] == "long"
