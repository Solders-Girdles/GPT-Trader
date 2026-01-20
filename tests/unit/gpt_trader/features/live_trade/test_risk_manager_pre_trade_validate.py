"""Tests for LiveRiskManager.pre_trade_validate."""

from __future__ import annotations

from decimal import Decimal

import pytest

from gpt_trader.features.live_trade.risk.manager import LiveRiskManager, ValidationError
from tests.unit.gpt_trader.features.live_trade.risk_manager_test_utils import (  # naming: allow
    MockConfig,  # naming: allow
)


@pytest.fixture(autouse=True)
def mock_load_state(monkeypatch: pytest.MonkeyPatch):
    """Prevent LiveRiskManager from loading state during tests."""
    monkeypatch.setattr(LiveRiskManager, "_load_state", lambda self: None)


class TestPreTradeValidate:
    """Tests for pre_trade_validate method."""

    def test_no_config_passes(self) -> None:
        """Test validation passes when no config."""
        manager = LiveRiskManager()

        result = manager.pre_trade_validate(
            symbol="BTC-USD",
            side="buy",
            quantity=Decimal("1"),
            price=Decimal("50000"),
            product=None,
            equity=Decimal("10000"),
            current_positions={},
        )
        assert result is None

    def test_leverage_within_limit(self) -> None:
        """Test validation passes when leverage within limit."""
        config = MockConfig(max_leverage=Decimal("10"))
        manager = LiveRiskManager(config=config)

        # Notional = 1 * 50000 = 50000, Leverage = 50000 / 10000 = 5x
        result = manager.pre_trade_validate(
            symbol="BTC-USD",
            side="buy",
            quantity=Decimal("1"),
            price=Decimal("50000"),
            product=None,
            equity=Decimal("10000"),
            current_positions={},
        )
        assert result is None

    def test_leverage_exceeds_limit_raises(self) -> None:
        """Test validation raises when leverage exceeds limit."""
        config = MockConfig(max_leverage=Decimal("5"))
        manager = LiveRiskManager(config=config)

        # Notional = 2 * 50000 = 100000, Leverage = 100000 / 10000 = 10x
        with pytest.raises(ValidationError, match="Leverage .* exceeds max 5"):
            manager.pre_trade_validate(
                symbol="BTC-USD",
                side="buy",
                quantity=Decimal("2"),
                price=Decimal("50000"),
                product=None,
                equity=Decimal("10000"),
                current_positions={},
            )

    def test_zero_equity_skips_validation(self) -> None:
        """Test validation skipped when equity is zero."""
        config = MockConfig(max_leverage=Decimal("5"))
        manager = LiveRiskManager(config=config)

        # Should not raise even with high notional
        result = manager.pre_trade_validate(
            symbol="BTC-USD",
            side="buy",
            quantity=Decimal("100"),
            price=Decimal("50000"),
            product=None,
            equity=Decimal("0"),
            current_positions={},
        )
        assert result is None

    def test_negative_equity_skips_validation(self) -> None:
        """Test validation skipped when equity is negative."""
        config = MockConfig(max_leverage=Decimal("5"))
        manager = LiveRiskManager(config=config)

        # Negative equity means the condition equity > 0 fails
        result = manager.pre_trade_validate(
            symbol="BTC-USD",
            side="buy",
            quantity=Decimal("100"),
            price=Decimal("50000"),
            product=None,
            equity=Decimal("-1000"),
            current_positions={},
        )
        assert result is None
