"""Tests for mark staleness guard integration."""

from __future__ import annotations


class TestMarkStalenessGuard:
    """Tests for mark staleness guard integration."""

    def test_check_mark_staleness_returns_true_when_no_update(self) -> None:
        """Test check_mark_staleness returns True when no mark update recorded."""
        from gpt_trader.features.live_trade.risk.config import RiskConfig
        from gpt_trader.features.live_trade.risk.manager import LiveRiskManager

        config = RiskConfig(daily_loss_limit_pct=0.05)
        manager = LiveRiskManager(config=config, state_file=None)

        # No mark update recorded - should be stale
        is_stale = manager.check_mark_staleness("BTC-USD")
        assert is_stale is True

    def test_check_mark_staleness_returns_false_when_fresh(self) -> None:
        """Test check_mark_staleness returns False when mark is fresh."""
        import time

        from gpt_trader.features.live_trade.risk.config import RiskConfig
        from gpt_trader.features.live_trade.risk.manager import LiveRiskManager

        config = RiskConfig(daily_loss_limit_pct=0.05)
        manager = LiveRiskManager(config=config, state_file=None)

        # Record a fresh mark update
        manager.last_mark_update["BTC-USD"] = time.time()

        is_stale = manager.check_mark_staleness("BTC-USD")
        assert is_stale is False

    def test_check_mark_staleness_returns_true_when_expired(self) -> None:
        """Test check_mark_staleness returns True when mark is too old."""
        import time

        from gpt_trader.features.live_trade.risk.config import RiskConfig
        from gpt_trader.features.live_trade.risk.manager import LiveRiskManager

        config = RiskConfig(daily_loss_limit_pct=0.05)
        manager = LiveRiskManager(config=config, state_file=None)

        # Record an old mark update (3 minutes ago, threshold is 2 minutes)
        manager.last_mark_update["BTC-USD"] = time.time() - 180

        is_stale = manager.check_mark_staleness("BTC-USD")
        assert is_stale is True
