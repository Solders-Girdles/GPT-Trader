"""Runtime guard tests for LiveRiskManager."""

import pytest
from decimal import Decimal
from datetime import datetime, timedelta

from bot_v2.features.live_trade.risk import LiveRiskManager
from bot_v2.config.live_trade_config import RiskConfig


class TestMarkStaleness:
    """Short-term mark price staleness checks."""

    @pytest.fixture
    def risk_manager(self):
        config = RiskConfig(max_mark_staleness_seconds=180)
        return LiveRiskManager(config=config)

    def test_fresh_mark_detection(self, risk_manager):
        is_stale = risk_manager.check_mark_staleness(
            symbol="BTC-PERP",
            mark_timestamp=datetime.utcnow(),
        )
        assert is_stale is False

    def test_stale_mark_detection(self, risk_manager):
        is_stale = risk_manager.check_mark_staleness(
            symbol="BTC-PERP",
            mark_timestamp=datetime.utcnow() - timedelta(minutes=7),
        )
        assert is_stale is True


class TestVolatilityCircuitBreaker:
    """Volatility guard behavior across thresholds."""

    @pytest.fixture
    def risk_manager(self):
        config = RiskConfig(
            enable_volatility_circuit_breaker=True,
            volatility_warning_threshold=0.15,
            volatility_reduce_only_threshold=0.20,
            volatility_kill_switch_threshold=0.25,
        )
        return LiveRiskManager(config=config)

    def test_low_volatility_allows_trading(self, risk_manager):
        base = Decimal("50000")
        marks = [base + Decimal(i) for i in range(25)]
        result = risk_manager.check_volatility_circuit_breaker(
            symbol="BTC-PERP",
            recent_marks=marks,
        )
        assert result.get("triggered") is False

    def test_high_volatility_triggers_warning(self, risk_manager):
        base = Decimal("50000")
        swings = [Decimal("8000"), Decimal("-8000")] * 13
        marks = [base]
        for delta in swings:
            marks.append(marks[-1] + delta)

        result = risk_manager.check_volatility_circuit_breaker(
            symbol="BTC-PERP",
            recent_marks=marks,
        )
        assert result.get("triggered") is True
        assert result.get("action") in {"warning", "reduce_only", "kill_switch"}
