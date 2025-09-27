#!/usr/bin/env python3
"""
Quick position sizing helper for Coinbase perps.

Computes USD notional and quantity from:
- account equity
- risk % per trade
- stop distance %

Then quantizes to exchange specs using config/brokers/coinbase_perp_specs.yaml.

Usage examples:
  python scripts/tools/size_from_stop.py \
    --symbol BTC-PERP --mark 63000 \
    --equity 10000 --risk-pct 0.005 --stop-pct 0.004

  python scripts/tools/size_from_stop.py \
    --symbol SOL-PERP --mark 150 \
    --equity 2500 --risk-pct 0.003 --stop-pct 0.006 --fee-bps 8 --buffer 1.2
"""

from __future__ import annotations

import argparse
import json
from decimal import Decimal
from pathlib import Path
import os

from bot_v2.features.brokerages.coinbase.specs import SpecsService


def main() -> int:
    ap = argparse.ArgumentParser(description="Position size from stop distance (Coinbase perps)")
    ap.add_argument("--symbol", required=True, help="Symbol, e.g., BTC-PERP")
    ap.add_argument("--mark", type=float, required=True, help="Current mark price")
    ap.add_argument("--equity", type=float, required=True, help="Account equity in USD")
    ap.add_argument("--risk-pct", type=float, required=True, help="Risk per trade as fraction, e.g., 0.005 for 0.5%")
    ap.add_argument("--stop-pct", type=float, required=True, help="Stop distance as fraction, e.g., 0.004 for 0.4%")
    ap.add_argument("--fee-bps", type=float, default=None, help="Roundtrip fee in bps (overrides env)")
    ap.add_argument("--buffer", type=float, default=1.25, help="Safety buffer multiplier on stop distance, default 1.25")
    ap.add_argument("--json", action="store_true", help="Output JSON only")
    args = ap.parse_args()

    equity = Decimal(str(args.equity))
    risk_pct = Decimal(str(args.risk_pct))
    stop_pct = Decimal(str(args.stop_pct))
    # Fee selection: env FEE_BPS_BY_SYMBOL or CLI flag fallback
    env_fee_map = {}
    try:
        env_map = os.getenv("FEE_BPS_BY_SYMBOL", "")
        if env_map:
            for token in env_map.split(","):
                if ":" in token:
                    k, v = token.split(":", 1)
                    env_fee_map[k.strip().upper()] = Decimal(str(v.strip()))
    except Exception:
        env_fee_map = {}

    fee_bps = env_fee_map.get(symbol)
    if fee_bps is None:
        fee_bps = Decimal(str(args.fee_bps if args.fee_bps is not None else 6.0))
    fee = fee_bps / Decimal("10000")
    buffer = Decimal(str(args.buffer))
    mark = Decimal(str(args.mark))
    symbol = args.symbol.upper()

    effective_stop = stop_pct * buffer + fee
    if effective_stop <= 0:
        raise SystemExit("Effective stop must be > 0")

    risk_usd = equity * risk_pct
    notional = risk_usd / effective_stop
    raw_qty = notional / mark

    # Quantize via SpecsService (loads config/brokers/coinbase_perp_specs.yaml)
    specs = SpecsService()
    size, reason = specs.calculate_safe_position_size(symbol, float(notional), float(mark))

    out = {
        "symbol": symbol,
        "mark": str(mark),
        "equity": str(equity),
        "risk_pct": str(risk_pct),
        "stop_pct": str(stop_pct),
        "buffer": str(buffer),
        "fee_bps": str(fee_bps),
        "effective_stop": str(effective_stop),
        "risk_usd": str(risk_usd),
        "target_notional_usd": str(notional.quantize(Decimal("0.01"))),
        "raw_qty": str(raw_qty),
        "quantized_qty": str(size),
        "quantization_reason": reason,
    }

    if args.json:
        print(json.dumps(out))
    else:
        print(f"Symbol:           {symbol}")
        print(f"Mark:             {mark}")
        print(f"Equity (USD):     {equity}")
        print(f"Risk %:           {risk_pct}  → Risk USD: {risk_usd}")
        print(f"Stop %:           {stop_pct}  Buffer: {buffer}  Fee bps: {args.fee_bps}")
        print(f"Effective stop:   {effective_stop:.6f}")
        print(f"Target notional:  ${out['target_notional_usd']}")
        print(f"Raw qty:          {raw_qty}")
        print(f"Quantized qty:    {size}  ({reason})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
