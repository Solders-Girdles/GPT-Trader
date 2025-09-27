#!/usr/bin/env python3
"""
Perps Preflight – readiness checks before going live.

Default runs offline (no network). Use --online to include light connectivity checks.

Checks:
- Environment variables for Coinbase perps
- Risk configuration sanity
- Presence of critical docs & profiles
- Optional: construct brokerage (no order placement)
"""

from __future__ import annotations

import os
import sys
import argparse
from pathlib import Path
from dataclasses import dataclass

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@dataclass
class CheckResult:
    ok: bool
    msg: str


def ok(msg: str) -> CheckResult:
    return CheckResult(True, f"✓ {msg}")


def bad(msg: str) -> CheckResult:
    return CheckResult(False, f"✗ {msg}")


def check_env() -> list[CheckResult]:
    out: list[CheckResult] = []
    broker = os.getenv("BROKER", "").lower()
    if broker == "coinbase":
        out.append(ok("BROKER=coinbase"))
    else:
        out.append(bad("BROKER must be 'coinbase' for perps"))

    perps_en = os.getenv("COINBASE_ENABLE_DERIVATIVES", "0") == "1"
    out.append(ok("COINBASE_ENABLE_DERIVATIVES=1" if perps_en else "COINBASE_ENABLE_DERIVATIVES not enabled"))

    sandbox = os.getenv("COINBASE_SANDBOX", "0") == "1"
    if perps_en and sandbox:
        out.append(bad("COINBASE_SANDBOX=1 with derivatives – perps require production (set 0)"))
    else:
        out.append(ok(f"COINBASE_SANDBOX={'1' if sandbox else '0'}"))

    # Advanced Trade JWT for production perps
    api_mode = os.getenv("COINBASE_API_MODE", "advanced").lower()
    out.append(ok(f"COINBASE_API_MODE={api_mode}"))
    if perps_en and not sandbox and api_mode == "advanced":
        cdp_key = bool(os.getenv("COINBASE_PROD_CDP_API_KEY")) or bool(os.getenv("COINBASE_CDP_API_KEY"))
        cdp_priv = bool(os.getenv("COINBASE_PROD_CDP_PRIVATE_KEY")) or bool(os.getenv("COINBASE_CDP_PRIVATE_KEY"))
        if not (cdp_key and cdp_priv):
            out.append(bad("Missing Advanced Trade JWT keys for production (COINBASE_PROD_CDP_API_KEY/PRIVATE_KEY)"))
        else:
            out.append(ok("Advanced Trade JWT present"))

    return out


def check_risk_config() -> list[CheckResult]:
    out: list[CheckResult] = []
    try:
        from bot_v2.config.live_trade_config import RiskConfig
        cfg = RiskConfig.from_env()
    except Exception as e:
        return [bad(f"RiskConfig load failed: {e}")]

    # Basic safety suggestions
    if cfg.max_leverage > 5:
        out.append(bad(f"RISK_MAX_LEVERAGE too high: {cfg.max_leverage}"))
    else:
        out.append(ok(f"RISK_MAX_LEVERAGE={cfg.max_leverage}"))

    if cfg.max_exposure_pct > 0.8:
        out.append(bad(f"RISK_MAX_EXPOSURE_PCT too high: {cfg.max_exposure_pct:.2f}"))
    else:
        out.append(ok(f"RISK_MAX_EXPOSURE_PCT={cfg.max_exposure_pct:.2f}"))

    if cfg.max_position_pct_per_symbol > 0.5:
        out.append(bad(f"RISK_MAX_POSITION_PCT_PER_SYMBOL too high: {cfg.max_position_pct_per_symbol:.2f}"))
    else:
        out.append(ok(f"RISK_MAX_POSITION_PCT_PER_SYMBOL={cfg.max_position_pct_per_symbol:.2f}"))

    out.append(ok(f"RISK_DAILY_LOSS_LIMIT={cfg.daily_loss_limit}"))
    out.append(ok(f"RISK_SLIPPAGE_GUARD_BPS={cfg.slippage_guard_bps}"))
    out.append(ok(f"RISK_MAX_MARK_STALENESS_SECONDS={cfg.max_mark_staleness_seconds}"))

    # Flags
    if cfg.reduce_only_mode:
        out.append(ok("Reduce-only mode enabled"))
    else:
        out.append(ok("Reduce-only mode disabled (ok for non-canary)"))
    if cfg.kill_switch_enabled:
        out.append(ok("Kill switch enabled (will block new orders)"))
    else:
        out.append(ok("Kill switch disabled (ensure manual procedure ready)"))

    # Time-of-day leverage schedule sanity
    if cfg.daytime_start_utc and cfg.daytime_end_utc:
        out.append(ok(f"Day window set: {cfg.daytime_start_utc}-{cfg.daytime_end_utc} UTC"))
        wanted = {"BTC-PERP", "ETH-PERP", "SOL-PERP", "XRP-PERP"}
        day_caps = set((cfg.day_leverage_max_per_symbol or {}).keys())
        night_caps = set((cfg.night_leverage_max_per_symbol or {}).keys())
        missing_day = wanted - day_caps
        missing_night = wanted - night_caps
        if missing_day:
            out.append(bad(f"Missing day leverage caps for: {sorted(missing_day)}"))
        if missing_night:
            out.append(bad(f"Missing night leverage caps for: {sorted(missing_night)}"))
    else:
        out.append(ok("No day/night window configured (global caps apply)"))

    return out


def check_docs() -> list[CheckResult]:
    out: list[CheckResult] = []
    required = [
        REPO_ROOT / "docs" / "RUNBOOK_PERPS.md",
        REPO_ROOT / "docs" / "GO_LIVE_CHECKLIST.md",
        REPO_ROOT / "docs" / "TRADING_PLAN_TEMPLATE.md",
        REPO_ROOT / "config" / "profiles" / "canary.yaml",
    ]
    for p in required:
        if p.exists():
            out.append(ok(f"Found {p.relative_to(REPO_ROOT)}"))
        else:
            out.append(bad(f"Missing {p.relative_to(REPO_ROOT)}"))
    return out


def check_connectivity() -> list[CheckResult]:
    out: list[CheckResult] = []
    try:
        from bot_v2.orchestration.broker_factory import create_brokerage
        broker = create_brokerage()
        # Avoid hitting private endpoints; just report base URLs and mode
        cfg = getattr(broker, "api_config", None)
        if cfg:
            out.append(ok(f"Broker created: mode={cfg.api_mode} sandbox={cfg.sandbox} ws={bool(cfg.ws_url)}"))
        else:
            out.append(ok("Broker created"))
    except Exception as e:
        out.append(bad(f"Broker creation failed: {e}"))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Perps preflight readiness checks")
    parser.add_argument("--online", action="store_true", help="Include online connectivity checks")
    args = parser.parse_args()

    if load_dotenv:
        load_dotenv(override=True)

    results: list[CheckResult] = []
    results += check_env()
    results += check_risk_config()
    results += check_docs()
    if args.online:
        results += check_connectivity()

    failures = [r for r in results if not r.ok]
    for r in results:
        print(r.msg)

    if failures:
        print(f"\nPreflight complete with {len(failures)} issue(s). Address before live trading.")
        return 2
    else:
        print("\nPreflight complete – all checks passed.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
