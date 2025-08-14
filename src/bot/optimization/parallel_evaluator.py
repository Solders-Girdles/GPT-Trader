"""
Standalone parallel evaluation functions for optimization.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


def evaluate_parameters_standalone(
    params: dict[str, Any], config_data: dict[str, Any]
) -> dict[str, Any]:
    """Standalone function for evaluating parameters (can be pickled for parallel processing)."""
    try:
        # Import here to avoid pickling issues
        from bot.backtest.engine_portfolio import run_backtest
        from bot.portfolio.allocator import PortfolioRules
        from bot.strategy.trend_breakout import TrendBreakoutParams, TrendBreakoutStrategy
        from bot.utils.validation import DateValidator

        # Extract config data
        symbols = config_data["symbols"]
        start_date = config_data["start_date"]
        end_date = config_data["end_date"]

        # Create strategy
        strategy_params = TrendBreakoutParams(
            donchian_lookback=params.get("donchian_lookback", 55),
            atr_period=params.get("atr_period", 20),
            atr_k=params.get("atr_k", 2.0),
        )
        strategy = TrendBreakoutStrategy(strategy_params)

        # Create portfolio rules
        rules = PortfolioRules(
            per_trade_risk_pct=params.get("risk_pct", 0.5) / 100.0,
            atr_k=params.get("atr_k", 2.0),
            max_positions=10,
            cost_bps=5.0,
        )

        # Run backtest for each symbol
        all_results = []
        for symbol in symbols:
            try:
                result = run_backtest(
                    symbol=symbol,
                    symbol_list_csv=None,
                    start=DateValidator.validate_date(start_date),
                    end=DateValidator.validate_date(end_date),
                    strategy=strategy,
                    rules=rules,
                    write_trades_csv=False,
                    write_summary_csv=False,
                    write_portfolio_csv=False,
                    quiet_mode=True,
                    return_summary=True,
                    show_progress=False,
                    make_plot=False,
                )
                if result:
                    all_results.append(result)
            except Exception as e:
                logger.warning(f"Failed to backtest {symbol}: {e}")

        if not all_results:
            return {
                "params": params,
                "sharpe": float("-inf"),
                "cagr": float("-inf"),
                "max_drawdown": float("inf"),
                "n_trades": 0,
                "error": "No results",
            }

        # Aggregate results
        avg_sharpe = sum(r.get("sharpe", 0) for r in all_results) / len(all_results)
        avg_cagr = sum(r.get("cagr", 0) for r in all_results) / len(all_results)
        avg_max_dd = sum(r.get("max_drawdown", 0) for r in all_results) / len(all_results)
        total_trades = sum(r.get("n_trades", 0) for r in all_results)

        return {
            "params": params,
            "sharpe": avg_sharpe,
            "cagr": avg_cagr,
            "max_drawdown": avg_max_dd,
            "n_trades": total_trades,
            "n_symbols": len(all_results),
        }

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        return {
            "params": params,
            "sharpe": float("-inf"),
            "cagr": float("-inf"),
            "max_drawdown": float("inf"),
            "n_trades": 0,
            "error": str(e),
        }


def evaluate_batch_standalone(
    batch: list[dict[str, Any]], config_data: dict[str, Any]
) -> list[dict[str, Any]]:
    """Evaluate a batch of parameter combinations."""
    results = []
    for params in batch:
        result = evaluate_parameters_standalone(params, config_data)
        results.append(result)
    return results
