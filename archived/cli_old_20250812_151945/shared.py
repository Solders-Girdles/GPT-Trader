from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import pandas as pd


def _parse_date(s: str) -> datetime:
    from bot.utils.validation import DateValidator

    return DateValidator.validate_date(s)


def _ensure_run_dir(strategy: str, run_tag: str = "", run_dir: str = "") -> Path:
    from bot.utils.paths import PathUtils

    if run_dir:
        return PathUtils.ensure_directory(run_dir)

    base_path = PathUtils.get_data_directory() / "experiments" / strategy
    prefix = f"{strategy}_{run_tag}" if run_tag else f"{strategy}_run"
    return PathUtils.create_timestamped_directory(base_path, prefix)


def _read_universe_csv(path: str) -> list[str]:
    from bot.utils.validation import FileValidator, SymbolValidator

    # Validate file exists
    file_path = FileValidator.validate_file_path(path)

    # Read CSV and find symbol column
    df = pd.read_csv(file_path)
    symbol_columns = ["symbol", "ticker", "Symbol", "Ticker", "SYMBOL", "TICKER"]

    for col in symbol_columns:
        if col in df.columns:
            symbols = [str(s).strip().upper() for s in df[col].dropna().unique()]
            return SymbolValidator.validate_symbols(symbols)

    raise ValueError("Universe CSV must have 'symbol' or 'ticker' column")


def _guesstimate_universe_label(symbol: str | None, symbol_list: str | None) -> str:
    if symbol:
        return symbol.upper()
    if symbol_list:
        return Path(symbol_list).stem
    return "UNKN"


def _compose_bt_basename(
    strategy: str,
    sym_label: str,
    start_s: str,
    end_s: str,
    don: int | None,
    atr: int | None,
    k: float | None,
    risk_pct: float | None,
    cadence: str,
    regime_on: bool,
    regime_win: int | None,
) -> str:
    parts = [
        strategy,
        sym_label,
        f"{start_s}_{end_s}",
        f"don{don}" if don is not None else None,
        f"atr{atr}" if atr is not None else None,
        f"k{(k or 0):.2f}" if k is not None else None,
        f"r{(risk_pct or 0):.2f}" if risk_pct is not None else None,
        cadence,
        (
            f"reg{regime_win}"
            if regime_on and regime_win is not None
            else ("reg0" if regime_on else None)
        ),
    ]
    return "__".join([p for p in parts if p])


def _wait_for_summary_since(
    before: set[str], timeout: float = 3.0, poll: float = 0.05
) -> Path | None:
    outdir = Path("data/backtests")
    deadline = time.time() + timeout
    while time.time() < deadline:
        candidates = [p for p in outdir.glob("PORT_*_summary.csv") if str(p) not in before]
        if candidates:
            return max(candidates, key=lambda p: p.stat().st_mtime)
        time.sleep(poll)
    return None


def _save_manifest(
    run_dir: Path, strategy: str, args: argparse.Namespace, symbols: list[str]
) -> None:
    manifest = {
        "strategy": strategy,
        "args": vars(args),
        "symbols": symbols,
    }
    (run_dir / "run.json").write_text(json.dumps(manifest, indent=2, default=str))


def _mirror_backtest_triplet_from_summary(
    summary_path: Path, dest_dir: Path, base_name: str
) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    stem = summary_path.name.replace("_summary.csv", "")
    src_dir = summary_path.parent
    for suffix in [".csv", ".png", "_summary.csv", "_trades.csv"]:
        src = src_dir / f"{stem}{suffix}"
        if src.exists():
            dst = dest_dir / f"{base_name}{suffix}"
            dst.write_bytes(src.read_bytes())
