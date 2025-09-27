from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest
import yaml

from scripts.run_spot_profile import main as run_profile_main


@pytest.fixture
def sample_profile(tmp_path: Path) -> Path:
    profile = {
        "symbols": ["ABC-USD"],
        "strategy": {
            "abc": {
                "type": "ma",
                "short_window": 2,
                "long_window": 3,
                "volume_filter": {"window": 2, "multiplier": 1.1},
                "momentum_filter": None,
                "trend_filter": None,
            }
        },
        "risk": {"commission_bps": 0.0, "initial_cash": 1000},
    }
    path = tmp_path / "profile.yaml"
    with path.open("w") as fh:
        yaml.safe_dump(profile, fh)
    return path


@pytest.fixture
def sample_data(tmp_path: Path) -> Path:
    pytest.importorskip("pyarrow")
    timestamps = [datetime(2024, 1, 1, tzinfo=timezone.utc) + pd.Timedelta(minutes=i) for i in range(5)]
    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": [100, 101, 102, 103, 104],
            "high": [101, 102, 103, 104, 105],
            "low": [99, 100, 101, 102, 103],
            "close": [101, 102, 103, 104, 105],
            "volume": [10, 11, 12, 13, 14],
        }
    )
    data_root = tmp_path / "data"
    data_root.mkdir()
    symbol_dir = data_root / "ABC_USD"
    symbol_dir.mkdir()
    df.to_parquet(symbol_dir / "candles_1h.parquet", index=False)
    return data_root


def test_run_spot_profile_cli(sample_profile: Path, sample_data: Path, capsys):
    pytest.importorskip("pyarrow")
    argv = [str(sample_profile), "--data-root", str(sample_data)]
    run_profile_main(argv)
    captured = capsys.readouterr()
    assert "ABC-USD" in captured.out
