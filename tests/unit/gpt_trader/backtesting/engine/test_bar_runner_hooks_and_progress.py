"""Tests for bar runner hooks and progress helpers."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock

import pytest
from tests.unit.gpt_trader.backtesting.engine.bar_runner_test_utils import (  # naming: allow
    _create_mock_data_provider,
)

from gpt_trader.backtesting.engine.bar_runner import ClockedBarRunner


class TestBarRunnerHooks:
    """Tests for event hooks."""

    def test_on_bar_start_adds_callback(self) -> None:
        provider = _create_mock_data_provider()
        runner = ClockedBarRunner(
            data_provider=provider,
            symbols=["BTC-USD"],
            granularity="ONE_HOUR",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 2),
        )

        callback = MagicMock()
        runner.on_bar_start(callback)

        assert callback in runner._on_bar_start_hooks

    def test_on_bar_end_adds_callback(self) -> None:
        provider = _create_mock_data_provider()
        runner = ClockedBarRunner(
            data_provider=provider,
            symbols=["BTC-USD"],
            granularity="ONE_HOUR",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 2),
        )

        callback = MagicMock()
        runner.on_bar_end(callback)

        assert callback in runner._on_bar_end_hooks


class TestBarRunnerProgress:
    """Tests for progress tracking."""

    def test_progress_starts_at_zero(self) -> None:
        provider = _create_mock_data_provider()
        runner = ClockedBarRunner(
            data_provider=provider,
            symbols=["BTC-USD"],
            granularity="ONE_HOUR",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 2),
        )

        assert runner.progress_pct == pytest.approx(0.0, abs=0.1)

    def test_progress_100_when_duration_zero(self) -> None:
        provider = _create_mock_data_provider()
        start = datetime(2024, 1, 1, 12, 0, 0)
        runner = ClockedBarRunner(
            data_provider=provider,
            symbols=["BTC-USD"],
            granularity="ONE_HOUR",
            start_date=start,
            end_date=start,
        )

        assert runner.progress_pct == 100.0

    def test_bars_remaining_initial(self) -> None:
        provider = _create_mock_data_provider()
        runner = ClockedBarRunner(
            data_provider=provider,
            symbols=["BTC-USD"],
            granularity="ONE_HOUR",
            start_date=datetime(2024, 1, 1, 0, 0, 0),
            end_date=datetime(2024, 1, 1, 10, 0, 0),
        )

        assert runner.bars_remaining == 10

    def test_bars_remaining_zero_duration(self) -> None:
        provider = _create_mock_data_provider()
        start = datetime(2024, 1, 1, 12, 0, 0)
        runner = ClockedBarRunner(
            data_provider=provider,
            symbols=["BTC-USD"],
            granularity="ONE_HOUR",
            start_date=start,
            end_date=start,
        )

        assert runner.bars_remaining == 0
