import logging

import numpy as np
import pandas as pd

from bot_v2.features.optimize.backtester import run_backtest_local


def test_run_backtest_emits_summary(caplog):
    dates = pd.date_range("2024-01-01", periods=120, freq="D")
    data = pd.DataFrame({"close": np.linspace(100.0, 112.0, len(dates))}, index=dates)

    params = {"fast_period": 6, "slow_period": 18}
    logger_name = "bot_v2.features.optimize.backtester"

    with caplog.at_level(logging.INFO, logger=logger_name):
        metrics = run_backtest_local("SimpleMA", data, params)

    messages = [record.getMessage() for record in caplog.records if record.name == logger_name]

    assert metrics.total_trades >= 0
    assert any("Backtest complete" in message for message in messages)
    assert any("bars=" in message and "trades=" in message for message in messages)
