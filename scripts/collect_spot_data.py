#!/usr/bin/env python3
"""Download historical spot candles from Coinbase and store them as Parquet files."""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List
import urllib.parse
import urllib.request

import pandas as pd
from dotenv import load_dotenv

_GRANULARITY_SECONDS: Dict[str, int] = {
    "1m": 60,
    "5m": 300,
    "15m": 900,
    "1h": 3600,
    "6h": 6 * 3600,
    "24h": 86400,
    "1d": 86400,
}


@dataclass
class Candle:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Coinbase spot candles to Parquet.")
    parser.add_argument("symbols", nargs="+", help="Symbols like BTC-USD ETH-USD")
    parser.add_argument("--start", required=True, help="ISO8601 start time (UTC)")
    parser.add_argument("--end", required=True, help="ISO8601 end time (UTC)")
    parser.add_argument("--granularity", default="1h", choices=_GRANULARITY_SECONDS.keys())
    parser.add_argument("--chunk-size", type=int, default=300, help="Candles per request (~300 max recommended)")
    parser.add_argument("--output", default="data/spot_raw", help="Directory for parquet files")
    parser.add_argument("--append", action="store_true", help="Append to existing parquet files")
    parser.add_argument("--sleep", type=float, default=0.35, help="Seconds to sleep between API requests")
    return parser.parse_args()


def load_env() -> None:
    env_path = os.environ.get("SPOT_DATA_ENV", ".env")
    if Path(env_path).exists():
        load_dotenv(dotenv_path=env_path, override=True)


def ensure_output_path(base: Path, symbol: str, granularity: str) -> Path:
    target_dir = base / symbol.replace("-", "_")
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir / f"candles_{granularity}.parquet"


def to_iso(ts: datetime) -> str:
    return ts.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def fetch_batch(symbol: str, start: datetime, end: datetime, granularity_seconds: int) -> List[Candle]:
    base_url = "https://api.exchange.coinbase.com"
    params = {
        "start": to_iso(start),
        "end": to_iso(end),
        "granularity": str(granularity_seconds),
    }
    query = urllib.parse.urlencode(params)
    url = f"{base_url}/products/{symbol}/candles?{query}"
    headers = {
        "Accept": "application/json",
        "User-Agent": "GPT-Trader/1.0 (research dataset collector)",
    }
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request) as resp:
            payload = resp.read()
    except urllib.error.HTTPError as exc:  # type: ignore[attr-defined]
        raise RuntimeError(f"Failed to fetch candles: {exc.code} {exc.reason} ({url})") from exc
    raw = json.loads(payload)
    candles: List[Candle] = []
    for entry in raw:
        ts, low, high, open_, close, volume = entry
        candles.append(
            Candle(
                timestamp=datetime.fromtimestamp(ts, tz=timezone.utc),
                open=float(open_),
                high=float(high),
                low=float(low),
                close=float(close),
                volume=float(volume),
            )
        )
    return candles


def write_parquet(path: Path, rows: Iterable[Candle], append: bool) -> int:
    data = [
        {
            "timestamp": candle.timestamp,
            "open": candle.open,
            "high": candle.high,
            "low": candle.low,
            "close": candle.close,
            "volume": candle.volume,
        }
        for candle in rows
    ]
    if not data:
        return 0
    df = pd.DataFrame(data)
    df.sort_values("timestamp", inplace=True)
    df.drop_duplicates(subset="timestamp", inplace=True)
    if append and path.exists():
        existing = pd.read_parquet(path)
        combined = pd.concat([existing, df], ignore_index=True)
        combined.sort_values("timestamp", inplace=True)
        combined.drop_duplicates(subset="timestamp", inplace=True)
        combined.to_parquet(path, index=False)
        return len(df)
    df.to_parquet(path, index=False)
    return len(df)


def collect(symbol: str, start: datetime, end: datetime, granularity: str, chunk_size: int, sleep_time: float) -> List[Candle]:
    seconds = _GRANULARITY_SECONDS[granularity]
    candles: List[Candle] = []
    window = timedelta(seconds=seconds * chunk_size)
    window_start = start

    while window_start < end:
        window_end = min(window_start + window, end)
        batch = fetch_batch(symbol, window_start, window_end, seconds)
        if not batch:
            window_start = window_end + timedelta(seconds=seconds)
            continue
        batch.sort(key=lambda c: c.timestamp)
        candles.extend(batch)
        last_ts = batch[-1].timestamp
        window_start = last_ts + timedelta(seconds=seconds)
        time.sleep(sleep_time)
    return candles


def main() -> int:
    args = parse_args()
    load_env()

    start = datetime.fromisoformat(args.start.replace("Z", "+00:00")).astimezone(timezone.utc)
    end = datetime.fromisoformat(args.end.replace("Z", "+00:00")).astimezone(timezone.utc)
    if end <= start:
        raise SystemExit("--end must be after --start")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "granularity": args.granularity,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "files": [],
    }

    for symbol in args.symbols:
        candles = collect(symbol, start, end, args.granularity, args.chunk_size, args.sleep)
        file_path = ensure_output_path(output, symbol, args.granularity)
        rows_written = write_parquet(file_path, candles, append=args.append)
        manifest["files"].append({
            "symbol": symbol,
            "rows": rows_written,
            "path": str(file_path.resolve()),
        })

    manifest_path = output / "manifest.json"
    with manifest_path.open("w") as fh:
        json.dump(manifest, fh, indent=2)

    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
