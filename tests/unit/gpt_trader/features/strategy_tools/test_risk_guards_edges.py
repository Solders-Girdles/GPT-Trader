from __future__ import annotations

from decimal import Decimal

from gpt_trader.features.strategy_tools.guards import RiskGuards


def test_liquidation_guard_disabled() -> None:
    guards = RiskGuards(min_liquidation_buffer_pct=None)

    ok, reason = guards.check_liquidation_distance(
        entry_price=Decimal("100"),
        position_size=Decimal("1"),
        leverage=Decimal("10"),
        account_equity=Decimal("1000"),
    )

    assert ok is True
    assert reason == "Liquidation guard disabled"


def test_liquidation_guard_fails_on_tight_buffer() -> None:
    guards = RiskGuards(min_liquidation_buffer_pct=Decimal("15"))

    ok, reason = guards.check_liquidation_distance(
        entry_price=Decimal("100"),
        position_size=Decimal("1"),
        leverage=Decimal("10"),
        account_equity=Decimal("1000"),
    )

    assert ok is False
    assert "Too close to liquidation" in reason


def test_liquidation_guard_passes_on_safe_buffer() -> None:
    guards = RiskGuards(min_liquidation_buffer_pct=Decimal("15"))

    ok, reason = guards.check_liquidation_distance(
        entry_price=Decimal("100"),
        position_size=Decimal("1"),
        leverage=Decimal("2"),
        account_equity=Decimal("1000"),
    )

    assert ok is True
    assert "Safe liquidation distance" in reason


def test_slippage_guard_disabled() -> None:
    guards = RiskGuards(max_slippage_impact_bps=None)

    ok, reason = guards.check_slippage_impact(order_size=Decimal("1"), market_snapshot={})

    assert ok is True
    assert reason == "Slippage guard disabled"


def test_slippage_guard_missing_depth() -> None:
    guards = RiskGuards(max_slippage_impact_bps=Decimal("20"))

    ok, reason = guards.check_slippage_impact(order_size=Decimal("1"), market_snapshot={})

    assert ok is False
    assert "Insufficient market data" in reason


def test_slippage_guard_rejects_order_larger_than_l10() -> None:
    guards = RiskGuards(max_slippage_impact_bps=Decimal("20"))

    ok, reason = guards.check_slippage_impact(
        order_size=Decimal("200"),
        market_snapshot={"depth_l1": Decimal("50"), "depth_l10": Decimal("100")},
    )

    assert ok is False
    assert "Order too large" in reason


def test_slippage_guard_allows_l1_size() -> None:
    guards = RiskGuards(max_slippage_impact_bps=Decimal("20"))

    ok, reason = guards.check_slippage_impact(
        order_size=Decimal("10"),
        market_snapshot={"depth_l1": Decimal("100"), "depth_l10": Decimal("200")},
    )

    assert ok is True
    assert "Acceptable slippage" in reason


def test_slippage_guard_rejects_large_impact_between_l1_and_l10() -> None:
    guards = RiskGuards(max_slippage_impact_bps=Decimal("20"))

    ok, reason = guards.check_slippage_impact(
        order_size=Decimal("80"),
        market_snapshot={"depth_l1": Decimal("50"), "depth_l10": Decimal("100")},
    )

    assert ok is False
    assert "Estimated slippage too high" in reason
