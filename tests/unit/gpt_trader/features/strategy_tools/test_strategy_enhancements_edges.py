from __future__ import annotations

from decimal import Decimal

from gpt_trader.features.strategy_tools.enhancements import StrategyEnhancements


def test_calculate_rsi_returns_none_when_too_short() -> None:
    enhancements = StrategyEnhancements()
    prices = [Decimal("100"), Decimal("101")]

    assert enhancements.calculate_rsi(prices) is None


def test_calculate_rsi_returns_100_when_avg_loss_zero() -> None:
    enhancements = StrategyEnhancements()
    prices = [Decimal("1"), Decimal("2"), Decimal("3")]

    assert enhancements.calculate_rsi(prices, period=2) == Decimal("100")


def test_calculate_rsi_period_override_midpoint() -> None:
    enhancements = StrategyEnhancements()
    prices = [Decimal("1"), Decimal("2"), Decimal("1")]

    assert enhancements.calculate_rsi(prices, period=2) == Decimal("50")


def test_should_confirm_ma_crossover_disabled() -> None:
    enhancements = StrategyEnhancements(rsi_confirmation_enabled=False)

    ok, reason = enhancements.should_confirm_ma_crossover("buy", [Decimal("1")])

    assert ok is True
    assert reason == "RSI confirmation disabled"


def test_should_confirm_ma_crossover_insufficient_data() -> None:
    enhancements = StrategyEnhancements(rsi_confirmation_enabled=True)

    ok, reason = enhancements.should_confirm_ma_crossover("buy", [Decimal("1")])

    assert ok is False
    assert "Insufficient price data" in reason


def test_should_confirm_ma_crossover_buy_sell_rejections() -> None:
    enhancements = StrategyEnhancements(rsi_confirmation_enabled=True)

    ok, reason = enhancements.should_confirm_ma_crossover("buy", [Decimal("1")], rsi=Decimal("80"))
    assert ok is False
    assert "RSI too high for buy" in reason

    ok, reason = enhancements.should_confirm_ma_crossover("sell", [Decimal("1")], rsi=Decimal("20"))
    assert ok is False
    assert "RSI too low for sell" in reason


def test_should_confirm_ma_crossover_unknown_signal() -> None:
    enhancements = StrategyEnhancements(rsi_confirmation_enabled=True)

    ok, reason = enhancements.should_confirm_ma_crossover("hold", [Decimal("1")], rsi=Decimal("50"))

    assert ok is False
    assert reason == "Unknown MA signal: hold"
