#!/usr/bin/env python3
"""
Quick test of baseline functionality without pytest overhead.
Used to verify imports work before running full test suite.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_import(module_path, item_name, description):
    """Test a specific import."""
    try:
        module = __import__(module_path, fromlist=[item_name])
        getattr(module, item_name)
        print(f"✓ {description}")
        return True
    except Exception as e:
        print(f"✗ {description}: {e}")
        return False


def main():
    print("=== Quick Baseline Import Test ===")

    tests = [
        ("bot.config", "get_config", "Configuration system"),
        ("bot.config", "TradingConfig", "Trading config class"),
        ("bot.strategy.base", "Strategy", "Strategy base class"),
        ("bot.strategy.demo_ma", "DemoMAStrategy", "Demo MA strategy"),
        ("bot.strategy.trend_breakout", "TrendBreakoutStrategy", "Trend breakout strategy"),
        ("bot.dataflow.sources.yfinance_source", "YFinanceSource", "YFinance data source"),
        ("bot.indicators.atr", "atr", "ATR indicator"),
        ("bot.indicators.donchian", "donchian_channels", "Donchian channels"),
        ("bot.backtest.engine_portfolio", "PortfolioBacktestEngine", "Portfolio backtest engine"),
        ("bot.logging", "get_logger", "Logging system"),
        ("bot.portfolio.allocator", "PortfolioAllocator", "Portfolio allocator"),
        ("bot.dataflow.validate", "validate_ohlcv", "Data validation"),
    ]

    passed = 0
    total = len(tests)

    for module_path, item_name, description in tests:
        if test_import(module_path, item_name, description):
            passed += 1

    print("\n=== Results ===")
    print(f"Passed: {passed}/{total} ({passed/total*100:.1f}%)")

    if passed == total:
        print("✓ All critical imports work!")
        return 0
    else:
        print("✗ Some critical imports failing")
        return 1


if __name__ == "__main__":
    sys.exit(main())
