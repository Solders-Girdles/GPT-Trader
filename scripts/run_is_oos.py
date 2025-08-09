# scripts/run_is_oos.py
from __future__ import annotations

import argparse
import csv
import itertools
import re
import subprocess  # noqa: S404  (std library)
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

# -------- Grid definition --------


@dataclass(frozen=True)
class ParamPoint:
    atr_k: float
    risk_pct: float
    max_positions: int
    entry_confirm: int
    min_rebalance_pct: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "atr_k": self.atr_k,
            "risk_pct": self.risk_pct,
            "max_positions": self.max_positions,
            "entry_confirm": self.entry_confirm,
            "min_rebalance_pct": self.min_rebalance_pct,
        }


def build_grid() -> list[ParamPoint]:
    atr_ks: Sequence[float] = [2.3, 2.4, 2.5, 2.6, 2.7]
    risk_pcts: Sequence[float] = [0.35, 0.40, 0.45, 0.50]
    max_pos: Sequence[int] = [8, 10, 12]
    entry_confirms: Sequence[int] = [1, 2]
    min_reb: Sequence[float] = [0.0, 0.002]

    grid: list[ParamPoint] = []
    for ak, rp, mp, ec, mr in itertools.product(
        atr_ks, risk_pcts, max_pos, entry_confirms, min_reb
    ):
        grid.append(ParamPoint(ak, rp, mp, ec, mr))
    return grid


# -------- Backtest runner --------

SUMMARY_RE = re.compile(r"Summary saved to ([^\s]+_summary\.csv)")


def run_one(
    params: ParamPoint,
    start: str,
    end: str,
    env: dict[str, str] | None = None,
) -> dict[str, Any] | None:
    """
    Run one backtest via CLI, parse the summary file path from stdout, return metrics dict.

    Returns None if the summary file could not be found/parsed.
    """
    cmd = [
        "python",
        "-m",
        "bot.cli",
        "backtest",
        "--strategy",
        "trend_breakout",
        "--symbol-list",
        "data/universe/sp100.csv",
        "--start",
        start,
        "--end",
        end,
        "--regime",
        "on",
        "--exit-mode",
        "stop",
        "--cost-bps",
        "5",
        "--atr-k",
        f"{params.atr_k}",
        "--risk-pct",
        f"{params.risk_pct}",
        "--max-positions",
        f"{params.max_positions}",
        "--entry-confirm",
        f"{params.entry_confirm}",
        "--min-rebalance-pct",
        f"{params.min_rebalance_pct}",
    ]

    # We call a trusted local CLI; suppress S603/S607 for this known-safe usage.
    try:
        proc = subprocess.run(  # noqa: S603
            [sys.executable, "-m", "poetry", "run", *cmd],
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception as e:  # pragma: no cover
        print(f"[run_one] Subprocess failed: {e}", flush=True)
        return None

    out = (proc.stdout or "") + "\n" + (proc.stderr or "")
    m = SUMMARY_RE.search(out)
    if not m:
        print("[run_one] Could not find summary path in output.", flush=True)
        return None

    summary_path = m.group(1)
    try:
        sr = pd.read_csv(summary_path, header=None, index_col=0, squeeze=True)  # type: ignore[arg-type]
        d: dict[str, Any] = sr.to_dict()
    except Exception as e:  # pragma: no cover
        print(f"[run_one] Failed reading summary '{summary_path}': {e}", flush=True)
        return None

    # Stamp and params
    d["is_start"] = start
    d["is_end"] = end
    for k, v in params.as_dict().items():
        d[k] = v
    return d


# -------- I/O helpers --------


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_results(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_parent(path)
    if not rows:
        # Write an empty sheet with a minimal header so downstream tools don't crash.
        header = [
            "atr_k",
            "risk_pct",
            "max_positions",
            "entry_confirm",
            "min_rebalance_pct",
            "sharpe",
            "cagr",
            "total_return",
            "max_drawdown",
            "total_costs",
            "is_start",
            "is_end",
        ]
        with open(path, "w", newline="") as f:
            writer_empty: csv.DictWriter[str] = csv.DictWriter(f, fieldnames=header)
            writer_empty.writeheader()
        return

    # Union of keys across rows, stable in first-seen order.
    keys: list[str] = []
    for r in rows:
        for k in r.keys():
            if k not in keys:
                keys.append(k)

    with open(path, "w", newline="") as f:
        writer_full: csv.DictWriter[str] = csv.DictWriter(f, fieldnames=keys)
        writer_full.writeheader()
        writer_full.writerows(rows)


# -------- Main --------

MIN_SHARPE = 0.75
MAX_DD_CAP = 0.12


def main() -> None:
    parser = argparse.ArgumentParser(description="Run IS/OOS grid with sharding support")
    parser.add_argument("--is-start", default="2010-01-01")
    parser.add_argument("--is-end", default="2018-12-31")
    parser.add_argument("--oos-start", default="2019-01-01")
    parser.add_argument("--oos-end", default="2024-12-31")
    parser.add_argument("--chunks", type=int, default=1, help="Total number of shards")
    parser.add_argument("--chunk", type=int, default=0, help="Shard index (0-based)")
    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        help="Randomly downsample grid BEFORE sharding (0=off)",
    )
    args = parser.parse_args()

    # Build full grid
    full_grid = build_grid()

    # Optional downsample BEFORE sharding
    if args.sample and 0 < args.sample < len(full_grid):
        import random

        random.seed(42)
        full_grid = random.sample(full_grid, args.sample)

    chunks = max(1, int(args.chunks))
    chunk = int(args.chunk)
    if chunk < 0 or chunk >= chunks:
        raise SystemExit(f"--chunk must be in [0, {chunks-1}]")

    shard = [p for i, p in enumerate(full_grid) if i % chunks == chunk]

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_is = Path(f"data/opt/is_grid_{stamp}_part{chunk+1}-of-{chunks}.csv")

    print(
        f"Running in-sample grid: {len(shard)} permutations "
        f"(IS: {args.is_start} → {args.is_end}) | shard {chunk+1}/{chunks}",
        flush=True,
    )

    rows: list[dict[str, Any]] = []
    for idx, pp in enumerate(shard, 1):
        print(f"[{idx}/{len(shard)}] {pp.as_dict()}", flush=True)
        res = run_one(pp, args.is_start, args.is_end)
        if res is not None:
            rows.append(res)

    write_results(out_is, rows)
    print(f"\nWrote in-sample grid to {out_is}", flush=True)

    # Quick IS preview
    try:
        df_is = pd.read_csv(out_is)
        if df_is.empty:
            print("\n(IS results file is empty.)", flush=True)
        else:
            top_is = df_is[
                (df_is.get("sharpe", pd.Series(dtype=float)) >= MIN_SHARPE)
                & (df_is.get("max_drawdown", pd.Series(dtype=float)) <= MAX_DD_CAP)
            ].copy()

            if top_is.empty:
                print(
                    "\nNo configs passed the IS filter—consider loosening constraints "
                    "or widening the grid.",
                    flush=True,
                )
            else:
                print(
                    f"\n== Shard Top by Sharpe "
                    f"(Sharpe≥{MIN_SHARPE:.2f}, MaxDD≤{MAX_DD_CAP:.0%}) ==",
                    flush=True,
                )
                cols_order = [
                    "atr_k",
                    "risk_pct",
                    "max_positions",
                    "entry_confirm",
                    "min_rebalance_pct",
                    "sharpe",
                    "cagr",
                    "max_drawdown",
                    "total_costs",
                ]
                cols = [c for c in cols_order if c in top_is.columns]
                print(
                    top_is.sort_values("sharpe", ascending=False)[cols]
                    .head(20)
                    .to_string(index=False)
                )
    except Exception as e:  # pragma: no cover
        print(f"\n[IS preview skipped: {e}]", flush=True)


if __name__ == "__main__":
    main()
