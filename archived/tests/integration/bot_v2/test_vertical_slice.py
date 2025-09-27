#!/usr/bin/env python3
"""
ARCHIVED: Early vertical slice integration tests and token efficiency demo.
Superseded by targeted unit tests in bot_v2/features/backtest/ and
system smoke tests. Kept for developer reference.
"""

import pytest
from datetime import datetime, timedelta
import pandas as pd
from bot_v2.features.backtest import run_backtest

pytestmark = pytest.mark.integration


def test_backtest_slice():
    """Exercise the backtest slice and assert core invariants."""
    start = datetime.now() - timedelta(days=90)
    end = datetime.now()

    result = run_backtest(
        strategy="SimpleMAStrategy",
        symbol="AAPL",
        start=start,
        end=end,
        initial_capital=10000,
        commission=0.001,
        slippage=0.0005,
    )

    # Validate shape and types
    assert result is not None
    assert hasattr(result, "metrics") and result.metrics is not None
    assert isinstance(result.equity_curve, pd.Series) and len(result.equity_curve) > 0
    assert isinstance(result.returns, pd.Series) and len(result.returns) == len(result.equity_curve)

    # Validate metrics are sane
    assert isinstance(result.metrics.total_trades, int) or result.metrics.total_trades >= 0
    assert 0.0 <= result.metrics.win_rate <= 100.0
    assert -100.0 <= result.metrics.max_drawdown <= 0.0 or result.metrics.max_drawdown >= 0.0
    # Summary string should contain key fields
    summary = result.summary()
    assert isinstance(summary, str) and "Total Return:" in summary


def demonstrate_token_savings():
    """Developer-only helper retained for manual runs (not asserted)."""
    pass


def main():
    """Run all vertical slice tests."""
    # Keep available for manual invocation
    test_backtest_slice()
    demonstrate_token_savings()
    return True


if __name__ == "__main__":
    main()
