"""Regression tests for calculate_correlations error handling."""

from typing import Any

import pandas as pd

from bot_v2.features.analyze import analyze


class _FailingProvider:
    """Data provider stub that always fails."""

    def get_historical_data(self, symbol: str, period: str) -> pd.DataFrame:  # noqa: D401 - simple stub
        raise RuntimeError(f"boom for {symbol}:{period}")


def test_calculate_correlations_logs_provider_errors(monkeypatch, caplog):
    """The function should log provider failures without raising."""

    def fake_provider_factory() -> Any:
        return _FailingProvider()

    monkeypatch.setattr(analyze, "get_data_provider", fake_provider_factory)

    with caplog.at_level("WARNING"):
        frame = analyze.calculate_correlations(["BTC-USD"], 5)

    assert frame.empty
    assert any(
        "Failed to load historical data for BTC-USD" in message for message in caplog.messages
    )
