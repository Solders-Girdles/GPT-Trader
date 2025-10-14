from datetime import datetime, timedelta

from unittest.mock import Mock

from bot_v2.features.brokerages.coinbase.adapter import CoinbaseBrokerage
from bot_v2.orchestration.configuration import BotConfig, Profile
from bot_v2.orchestration.perps_bot_builder import create_perps_bot
from bot_v2.orchestration.service_registry import ServiceRegistry


def test_consecutive_failures(monkeypatch):
    # Disable WS threads for test
    monkeypatch.setenv("DISABLE_WS_STREAMING", "1")
    config = BotConfig.from_profile(Profile.DEV.value)
    broker = Mock(spec=CoinbaseBrokerage)
    broker.__class__ = CoinbaseBrokerage
    registry = ServiceRegistry(config=config, broker=broker)
    bot = create_perps_bot(config, registry=registry)
    monitor = bot._market_monitor

    # First 3 failures shouldn't trigger fallback
    monitor.consecutive_failures = 0
    monitor.max_failures = 3
    monitor.record_failure()
    monitor.record_failure()
    monitor.record_failure()
    assert monitor.should_fallback_to_rest() is False

    # 4th failure triggers fallback
    monitor.record_failure()
    assert monitor.should_fallback_to_rest() is True


def test_staleness_detection(monkeypatch):
    monkeypatch.setenv("DISABLE_WS_STREAMING", "1")
    config = BotConfig.from_profile(Profile.DEV.value)
    broker = Mock(spec=CoinbaseBrokerage)
    broker.__class__ = CoinbaseBrokerage
    registry = ServiceRegistry(config=config, broker=broker)
    bot = create_perps_bot(config, registry=registry)
    monitor = bot._market_monitor
    monitor.max_staleness_ms = 1000

    # Fresh update should be fine
    monitor.record_update("BTC-PERP")
    assert monitor.check_staleness() is False

    # Backdate last update beyond threshold
    monitor.last_update["BTC-PERP"] = datetime.utcnow() - timedelta(milliseconds=1500)
    assert monitor.check_staleness() is True
    assert monitor.should_fallback_to_rest() is True
