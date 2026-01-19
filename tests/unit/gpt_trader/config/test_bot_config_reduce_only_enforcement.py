"""Tests for reduce-only mode enforcement in order flow."""

from __future__ import annotations


class TestReduceOnlyEnforcement:
    """Tests for reduce-only mode enforcement in order flow."""

    def test_check_order_blocks_new_position_in_reduce_only(self) -> None:
        """Test check_order blocks new positions when in reduce-only mode."""
        from gpt_trader.features.live_trade.risk.config import RiskConfig
        from gpt_trader.features.live_trade.risk.manager import LiveRiskManager

        config = RiskConfig(daily_loss_limit_pct=0.05)
        manager = LiveRiskManager(config=config, state_file=None)

        # Manually trigger reduce-only mode
        manager.set_reduce_only_mode(True, reason="test")

        # Non-reduce-only order should be blocked
        order = {"symbol": "BTC-USD", "side": "BUY", "reduce_only": False}
        assert manager.check_order(order) is False

    def test_check_order_allows_reduce_only_order(self) -> None:
        """Test check_order allows reduce-only orders when in reduce-only mode."""
        from gpt_trader.features.live_trade.risk.config import RiskConfig
        from gpt_trader.features.live_trade.risk.manager import LiveRiskManager

        config = RiskConfig(daily_loss_limit_pct=0.05)
        manager = LiveRiskManager(config=config, state_file=None)

        # Manually trigger reduce-only mode
        manager.set_reduce_only_mode(True, reason="test")

        # Reduce-only order should be allowed
        order = {"symbol": "BTC-USD", "side": "SELL", "reduce_only": True}
        assert manager.check_order(order) is True

    def test_daily_loss_triggers_reduce_only(self) -> None:
        """Test daily loss limit triggers reduce-only mode."""
        from decimal import Decimal

        from gpt_trader.features.live_trade.risk.config import RiskConfig
        from gpt_trader.features.live_trade.risk.manager import LiveRiskManager

        config = RiskConfig(daily_loss_limit_pct=0.05)
        manager = LiveRiskManager(config=config, state_file=None)

        # Set start equity
        manager.track_daily_pnl(Decimal("10000"), {})

        # 6% loss should trigger (> 5% limit)
        triggered = manager.track_daily_pnl(Decimal("9400"), {})

        assert triggered is True
        assert manager._reduce_only_mode is True
        assert manager._reduce_only_reason == "daily_loss_limit_breached"
