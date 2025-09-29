import csv
from datetime import datetime, timezone
from decimal import Decimal

import pandas as pd
import pytest

from bot_v2.features.backtest.spot import load_candles_from_parquet


@pytest.fixture
def sample_parquet(tmp_path):
    ts = [datetime(2024, 1, 1, tzinfo=timezone.utc)]
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "open": [100],
            "high": [101],
            "low": [99],
            "close": [100],
            "volume": [1],
        }
    )
    path = tmp_path / "mini.parquet"
    df.to_parquet(path, index=False)
    return path


@pytest.mark.parametrize("strategy", ["ma", "bollinger"])
def test_sweep_script_smoke(tmp_path, strategy):
    pytest.importorskip("pyarrow")
    from scripts.sweep_strategies import main as sweep_main

    parquet_path = tmp_path / "candles.parquet"
    df = pd.DataFrame(
        {
            "timestamp": [
                datetime(2024, 1, 1, tzinfo=timezone.utc) + pd.Timedelta(hours=i) for i in range(30)
            ],
            "open": list(range(100, 130)),
            "high": [v + 1 for v in range(100, 130)],
            "low": [v - 1 for v in range(100, 130)],
            "close": list(range(100, 130)),
            "volume": [1 for _ in range(30)],
        }
    )
    df.to_parquet(parquet_path, index=False)

    output = tmp_path / f"results_{strategy}.csv"
    argv_backup = list(__import__("sys").argv)
    try:
        __import__("sys").argv = [
            "sweep",
            str(parquet_path),
            "--strategy",
            strategy,
            "--short",
            "5",
            "--long",
            "10",
            "--boll-window",
            "5",
            "--boll-std",
            "1.5",
            "--output",
            str(output),
        ]
        sweep_main()
    finally:
        __import__("sys").argv = argv_backup

    assert output.exists()
    with output.open() as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
        assert len(rows) >= 1
        assert "total_return" in rows[0]
