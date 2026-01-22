"""Tests for LiveRiskManager protection: liquidation buffer and reduce-only mode."""

from __future__ import annotations

from decimal import Decimal

import pytest

from gpt_trader.features.live_trade.risk.manager import LiveRiskManager
from tests.unit.gpt_trader.features.live_trade.risk_manager_test_utils import (  # naming: allow
    MockConfig,
    MockPosition,
)


@pytest.fixture(autouse=True)
def mock_load_state(monkeypatch: pytest.MonkeyPatch):
    """Prevent LiveRiskManager from loading state during tests."""
    monkeypatch.setattr(LiveRiskManager, "_load_state", lambda self: None)


class TestReduceOnlyMode:
    """Tests for reduce-only mode methods."""

    def test_default_not_reduce_only(self) -> None:
        """Should default to not reduce-only."""
        manager = LiveRiskManager()

        assert manager.is_reduce_only_mode() is False

    def test_set_reduce_only_true(self) -> None:
        """Should set reduce-only to True."""
        manager = LiveRiskManager()

        manager.set_reduce_only_mode(True)

        assert manager.is_reduce_only_mode() is True

    def test_set_reduce_only_false(self) -> None:
        """Should set reduce-only to False."""
        manager = LiveRiskManager()
        manager._reduce_only_mode = True

        manager.set_reduce_only_mode(False)

        assert manager.is_reduce_only_mode() is False

    def test_set_reduce_only_with_reason(self) -> None:
        """Should accept reason when setting reduce-only."""
        manager = LiveRiskManager()

        manager.set_reduce_only_mode(True, reason="liquidation_warning")

        assert manager._reduce_only_mode is True
        assert manager._reduce_only_reason == "liquidation_warning"

    def test_set_reduce_only_clears_reason_when_false(self) -> None:
        """Should clear reason when disabling reduce-only."""
        manager = LiveRiskManager()
        manager._reduce_only_mode = True
        manager._reduce_only_reason = "some_reason"

        manager.set_reduce_only_mode(False, reason="")

        assert manager._reduce_only_mode is False
        assert manager._reduce_only_reason == ""


class TestCheckLiquidationBuffer:
    """Tests for check_liquidation_buffer method."""

    def test_no_config_returns_false(self) -> None:
        """Should return False when no config."""
        manager = LiveRiskManager()

        result = manager.check_liquidation_buffer(
            "BTC-USD", {"liquidation_price": 100, "mark": 110}, Decimal("1000")
        )

        assert result is False

    def test_dict_position_no_liquidation_price(self) -> None:
        """Should return False when liquidation_price missing."""
        config = MockConfig()
        manager = LiveRiskManager(config=config)

        result = manager.check_liquidation_buffer("BTC-USD", {"mark": 110}, Decimal("1000"))

        assert result is False

    def test_dict_position_no_mark_price(self) -> None:
        """Should return False when mark price missing."""
        config = MockConfig()
        manager = LiveRiskManager(config=config)

        result = manager.check_liquidation_buffer(
            "BTC-USD", {"liquidation_price": 100}, Decimal("1000")
        )

        assert result is False

    def test_dict_position_with_mark_key(self) -> None:
        """Should work with 'mark' key in dict."""
        config = MockConfig(min_liquidation_buffer_pct=Decimal("0.05"))
        manager = LiveRiskManager(config=config)

        # Buffer = |110 - 100| / 110 = 0.0909 (9.09%), threshold 5%
        result = manager.check_liquidation_buffer(
            "BTC-USD", {"liquidation_price": 100, "mark": 110}, Decimal("1000")
        )

        assert result is False  # Buffer > threshold, not dangerous

    def test_dict_position_with_mark_price_key(self) -> None:
        """Should work with 'mark_price' key in dict."""
        config = MockConfig(min_liquidation_buffer_pct=Decimal("0.05"))
        manager = LiveRiskManager(config=config)

        result = manager.check_liquidation_buffer(
            "BTC-USD", {"liquidation_price": 100, "mark_price": 110}, Decimal("1000")
        )

        assert result is False

    def test_object_position_with_mark_price(self) -> None:
        """Should work with object having mark_price attribute."""
        config = MockConfig(min_liquidation_buffer_pct=Decimal("0.05"))
        manager = LiveRiskManager(config=config)
        position = MockPosition(liquidation_price=Decimal("100"), mark_price=Decimal("110"))

        result = manager.check_liquidation_buffer("BTC-USD", position, Decimal("1000"))

        assert result is False

    def test_object_position_with_mark_fallback(self) -> None:
        """Should fall back to mark attribute."""
        config = MockConfig(min_liquidation_buffer_pct=Decimal("0.05"))
        manager = LiveRiskManager(config=config)
        position = MockPosition(liquidation_price=Decimal("100"), mark=Decimal("110"))

        result = manager.check_liquidation_buffer("BTC-USD", position, Decimal("1000"))

        assert result is False

    def test_triggers_reduce_only_when_buffer_too_small(self) -> None:
        """Should trigger reduce_only when buffer below threshold."""
        config = MockConfig(min_liquidation_buffer_pct=Decimal("0.15"))
        manager = LiveRiskManager(config=config)

        # Buffer = |110 - 100| / 110 = 0.0909 (9.09%), threshold 15%
        result = manager.check_liquidation_buffer(
            "BTC-USD", {"liquidation_price": 100, "mark": 110}, Decimal("1000")
        )

        assert result is True
        assert manager.positions["BTC-USD"]["reduce_only"] is True

    def test_liquidation_buffer_triggers_reduce_only(self) -> None:
        """Should trigger reduce-only for symbol when buffer low."""
        config = MockConfig(min_liquidation_buffer_pct=Decimal("0.20"))
        manager = LiveRiskManager(config=config)

        # Buffer = |105 - 100| / 105 ≈ 4.76%, threshold 20%
        triggered = manager.check_liquidation_buffer(
            "BTC-PERP",
            {"liquidation_price": 100, "mark": 105},
            Decimal("1000"),
        )

        assert triggered is True
        assert manager.positions["BTC-PERP"]["reduce_only"] is True

    def test_zero_mark_price_returns_false(self) -> None:
        """Should return False when mark price is zero."""
        config = MockConfig()
        manager = LiveRiskManager(config=config)

        result = manager.check_liquidation_buffer(
            "BTC-USD", {"liquidation_price": 100, "mark": 0}, Decimal("1000")
        )

        assert result is False

    def test_handles_attribute_error(self) -> None:
        """Should handle AttributeError gracefully."""
        config = MockConfig()
        manager = LiveRiskManager(config=config)

        result = manager.check_liquidation_buffer("BTC-USD", object(), Decimal("1000"))

        assert result is False

    def test_handles_none_values(self) -> None:
        """Should handle None values gracefully."""
        config = MockConfig()
        manager = LiveRiskManager(config=config)

        result = manager.check_liquidation_buffer(
            "BTC-USD", {"liquidation_price": None, "mark": 110}, Decimal("1000")
        )

        assert result is False

    def test_handles_value_error(self) -> None:
        """Should handle ValueError gracefully."""
        config = MockConfig()
        manager = LiveRiskManager(config=config)

        result = manager.check_liquidation_buffer(
            "BTC-USD", {"liquidation_price": "", "mark": 110}, Decimal("1000")
        )

        assert result is False
