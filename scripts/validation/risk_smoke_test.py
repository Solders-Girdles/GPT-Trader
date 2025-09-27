#!/usr/bin/env python3
"""
Risk Smoke Test

Validates simplified risk behavior:
- Prints current RiskConfig for sanity
- Staleness behavior: soft warning vs hard halt
- Volatility circuit breaker: off by default; exercise thresholds when enabled

Run:
  python scripts/validation/risk_smoke_test.py
"""

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal

from bot_v2.config.live_trade_config import RiskConfig
from bot_v2.features.live_trade.risk import LiveRiskManager


def print_config(cfg: RiskConfig) -> None:
    print("== RiskConfig ==")
    print(f"max_leverage: {cfg.max_leverage}")
    print(f"max_exposure_pct: {cfg.max_exposure_pct}")
    print(f"max_position_pct_per_symbol: {cfg.max_position_pct_per_symbol}")
    print(f"min_liquidation_buffer_pct: {cfg.min_liquidation_buffer_pct}")
    print(f"slippage_guard_bps: {cfg.slippage_guard_bps}")
    print(f"max_mark_staleness_seconds (soft): {cfg.max_mark_staleness_seconds}")
    print(f"volatility_cb_enabled: {cfg.enable_volatility_circuit_breaker}")
    print(f"vol thresholds (warn/reduce/kill): {cfg.volatility_warning_threshold}/"
          f"{cfg.volatility_reduce_only_threshold}/{cfg.volatility_kill_switch_threshold}")


def test_staleness(rm: LiveRiskManager, symbol: str = "BTC-PERP") -> None:
    print("\n== Staleness Behavior ==")
    soft = rm.config.max_mark_staleness_seconds
    now = datetime.utcnow()

    # Fresh (no halt)
    rm.last_mark_update[symbol] = now - timedelta(seconds=soft // 2)
    halt = rm.check_mark_staleness(symbol)
    print(f"fresh age ~{soft//2}s → halt? {halt}")

    # Soft-warning (continue)
    rm.last_mark_update[symbol] = now - timedelta(seconds=int(soft * 1.2))
    halt = rm.check_mark_staleness(symbol)
    print(f"slightly stale ~{int(soft*1.2)}s → halt? {halt} (expect False)")

    # Hard-halt (>2x)
    rm.last_mark_update[symbol] = now - timedelta(seconds=int(soft * 2.5))
    halt = rm.check_mark_staleness(symbol)
    print(f"severely stale ~{int(soft*2.5)}s → halt? {halt} (expect True)")


def test_volatility_cb(rm: LiveRiskManager, symbol: str = "BTC-PERP") -> None:
    print("\n== Volatility Circuit Breaker ==")
    # Enable CB for the test (off by default)
    rm.config.enable_volatility_circuit_breaker = True
    window = max(20, getattr(rm.config, 'volatility_window_periods', 20))

    # Low-vol series
    base = 100.0
    low_vol = [Decimal(str(base + i * 0.01)) for i in range(window)]  # ~0.01 increments
    res_low = rm.check_volatility_circuit_breaker(symbol, low_vol)
    print(f"low vol → triggered? {res_low.get('triggered')} action={res_low.get('action')} vol={res_low.get('volatility')}")

    # High-vol series (alternating ~±10%)
    hi = []
    px = Decimal('100')
    for i in range(window):
        px = px * (Decimal('1.10') if i % 2 == 0 else Decimal('0.90'))
        hi.append(px)
    res_hi = rm.check_volatility_circuit_breaker(symbol, hi)
    print(f"high vol → triggered? {res_hi.get('triggered')} action={res_hi.get('action')} vol={res_hi.get('volatility')}")


def main() -> int:
    cfg = RiskConfig.from_env()
    print_config(cfg)
    rm = LiveRiskManager(config=cfg)

    test_staleness(rm)
    test_volatility_cb(rm)
    print("\nRisk smoke test completed ✔︎")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

