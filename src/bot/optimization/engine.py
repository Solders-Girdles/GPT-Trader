"""
Optimization engine with rapid evolution improvements and major performance optimizations.
"""

from __future__ import annotations

import json
import logging
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from bot.utils.validation import DateValidator

from .analyzer import ResultAnalyzer
from .config import OptimizationConfig
from .evolutionary import EvolutionaryOptimizer
from .grid import GridOptimizer
from .strategy_diversity import StrategyDiversityTracker
from .visualizer import ResultVisualizer

logger = logging.getLogger(__name__)


def _convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: _convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [_convert_numpy_types(item) for item in obj]
    else:
        return obj


class OptimizationEngine:
    """Optimization engine with rapid evolution improvements and major performance optimizations."""

    def __init__(self, config: OptimizationConfig) -> None:
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize optimizers
        self.grid_optimizer = GridOptimizer(config) if config.grid_search else None
        self.evolutionary_optimizer = EvolutionaryOptimizer(config) if config.evolutionary else None

        # Initialize analysis tools
        self.analyzer = ResultAnalyzer()
        self.visualizer = ResultVisualizer()

        # Initialize diversity tracker
        self.diversity_tracker = StrategyDiversityTracker(
            self.output_dir, config.parameter_space.strategy
        )

        # Strategy factory
        self.strategy_factory = {
            "trend_breakout": self._create_trend_breakout_strategy,
            "demo_ma": self._create_demo_ma_strategy,
        }

        # Results storage
        self.results: list[dict[str, Any]] = []
        self.best_result: dict[str, Any] | None = None

        # Performance tracking for rapid evolution
        self.performance_history = {
            "best_sharpe": [],
            "avg_sharpe": [],
            "diversity_score": [],
            "generation": [],
        }

        # Performance optimizations
        self._setup_performance_optimizations()
        # Feature/signal cache for coarse vectorized runs
        self._signal_cache: dict[tuple[str, int, int], pd.DataFrame] = {}
        # Research features (from config.extra) for future model-guided flows
        self.research_features: list[str] = []
        try:
            extra = getattr(self.config, "extra", {})
            feats = extra.get("features", []) if isinstance(extra, dict) else []
            if isinstance(feats, list):
                self.research_features = [str(x) for x in feats]
        except Exception:
            self.research_features = []

    def _setup_performance_optimizations(self) -> None:
        """Setup performance optimizations."""
        # Pre-load data for all symbols to avoid repeated downloads
        self._preload_data()

        # Setup caching
        self._evaluation_cache = {}

        # Setup parallel processing
        self.max_workers = min(self.config.max_workers, mp.cpu_count())
        logger.info(f"Using {self.max_workers} workers for parallel processing")

    def _preload_data(self) -> None:
        """Pre-load market data for all symbols to avoid repeated downloads."""
        try:
            from bot.dataflow.sources.yfinance_source import YFinanceSource
            from bot.dataflow.validate import adjust_to_adjclose, validate_daily_bars

            logger.info("Pre-loading market data for performance optimization...")

            self.market_data = {}
            source = YFinanceSource()

            for symbol in self.config.symbols:
                try:
                    # Load and validate data (use existing YF source API)
                    df = source.get_daily_bars(
                        symbol,
                        start=self.config.start_date,
                        end=self.config.end_date,
                    )
                    df_adj, _ = adjust_to_adjclose(df)
                    validate_daily_bars(df_adj, symbol)
                    df_adj.index = pd.to_datetime(df_adj.index).tz_localize(None)

                    self.market_data[symbol] = df_adj
                    logger.debug(f"Pre-loaded data for {symbol}: {len(df)} rows")

                except Exception as e:
                    logger.warning(f"Failed to pre-load data for {symbol}: {e}")

            logger.info(f"Pre-loaded data for {len(self.market_data)} symbols")

        except Exception as e:
            logger.warning(f"Failed to pre-load data: {e}")
            self.market_data = {}

    def run(self) -> dict[str, Any]:
        """Run optimization with rapid evolution improvements and performance optimizations."""
        logger.info(f"Starting optimization: {self.config.name}")

        start_time = datetime.now()
        # Optional: load seeds at start
        try:
            self._maybe_load_seeds()
        except Exception as e:
            logger.warning(f"Seed load skipped: {e}")

        # Optional: coarse-then-refine pipeline
        if self.config.coarse_then_refine:
            logger.info("Running coarse-then-refine pipeline")
            coarse_results = self._run_coarse_stage()
            self.results.extend(coarse_results)
            refine_results = self._run_refine_stage(coarse_results)
            self.results.extend(refine_results)
        else:
            # Run grid search if enabled
            if self.grid_optimizer:
                logger.info("Running grid search...")
                grid_results = self._run_grid_search_optimized()
                self.results.extend(grid_results)
                logger.info(f"Grid search completed: {len(grid_results)} evaluations")

        # Run evolutionary optimization if enabled
        if self.evolutionary_optimizer:
            logger.info("Running evolutionary optimization...")
            evo_results = self._run_evolutionary_optimized()
            self.results.extend(evo_results)
            logger.info(f"Evolutionary optimization completed: {len(evo_results)} evaluations")

        # Find best result
        if self.results:
            self.best_result = max(self.results, key=lambda x: x.get("sharpe", float("-inf")))

        # Generate analysis and per-run HTML report
        if self.config.create_plots:
            self._generate_analysis()

        # Generate summary report first
        summary = self._generate_summary()

        try:
            from .report import create_run_report

            report_path = create_run_report(
                self.output_dir,
                self.results,
                getattr(self.evolutionary_optimizer, "generation_history", []),
                summary,
            )
            logger.info(f"Run report written to {report_path}")
        except Exception as e:
            logger.warning(f"Run report generation failed: {e}")
        summary = _convert_numpy_types(summary)  # Convert numpy types
        summary_path = self.output_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        evaluations_per_second = len(self.results) / duration if duration > 0 else 0

        logger.info(f"Optimization complete in {duration:.1f}s")
        logger.info(f"Evaluations per second: {evaluations_per_second:.1f}")
        logger.info(f"Results saved to {self.output_dir}")

        # Auto-write seeds.json (top-k)
        try:
            self._write_seeds(topk=max(1, int(self.config.seed_topk)))
        except Exception as e:
            logger.warning(f"Failed to write seeds.json: {e}")

        return summary

    def _run_coarse_stage(self) -> list[dict[str, Any]]:
        """Run a fast coarse stage on fewer symbols and shorter window."""
        # Build a symbol subset
        symbols_all = list(self.config.symbols)
        subset = symbols_all[: max(1, int(self.config.coarse_symbols))]
        # Shorten period
        end_dt = DateValidator.validate_date(self.config.end_date)
        start_dt = end_dt - timedelta(days=int(self.config.coarse_months * 30))
        # Prepare once
        from bot.backtest.engine_portfolio import prepare_backtest_data

        prepared = prepare_backtest_data(
            symbol=None,
            symbol_list_csv=None,
            start=start_dt,
            end=end_dt,
            regime_on=False,
            strict_mode=True,
            symbols=subset,
        )
        # Evaluate grid (or sampled grid) with vectorized fast flags
        combinations = self.config.parameter_space.get_grid_combinations()
        if (
            self.config.grid_sample_size
            and self.config.grid_sample_size > 0
            and len(combinations) > self.config.grid_sample_size
        ):
            combinations = combinations[: self.config.grid_sample_size]
        results: list[dict[str, Any]] = []
        # Optional progress bar for coarse phase
        pbar = None
        if not getattr(self.config, "quiet_bars", False):
            try:
                from tqdm import tqdm  # type: ignore

                pbar = tqdm(total=len(combinations), desc="Coarse", leave=True)
            except Exception:
                pbar = None
        # Evaluate in batches to amortize overhead
        batch = []
        for params in combinations:
            batch.append(params)
            if len(batch) >= 16:
                for p in batch:
                    results.append(
                        self._evaluate_params_with_prepared(p, prepared=prepared, phase1=True)
                    )
                if pbar is not None:
                    try:
                        pbar.update(len(batch))
                    except Exception:
                        pass
                batch = []
        for p in batch:
            results.append(self._evaluate_params_with_prepared(p, prepared=prepared, phase1=True))
        if pbar is not None:
            try:
                # Account for any remaining last batch
                pbar.update(len(batch))
                pbar.close()
            except Exception:
                pass
        # Successive halving style pruning; in vectorized phase1 we may not have reliable trade counts
        filtered = results
        if not self.config.vectorized_phase1:
            filtered = [r for r in filtered if r.get("n_trades", 0) >= self.config.min_trades]
        filtered = [r for r in filtered if r.get("sharpe", -1e9) >= self.config.min_sharpe]
        filtered = [r for r in filtered if r.get("max_drawdown", 1e9) <= self.config.max_drawdown]
        # Fallback: if filtering dropped everything, keep top 1% by Sharpe (at least 1)
        if not filtered and results:
            try:
                topn = max(1, int(len(results) * 0.01))
                filtered = sorted(
                    results, key=lambda r: r.get("sharpe", float("-inf")), reverse=True
                )[:topn]
                logger.info(
                    f"Coarse filtering yielded 0; falling back to top {topn} by Sharpe for refine stage"
                )
            except Exception:
                filtered = results[:1]
        return filtered

    def _run_refine_stage(self, coarse_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Refine top-k% parameters on full symbols/period with full fidelity."""
        if not coarse_results:
            return []
        # Select top fraction
        sorted_res = sorted(
            coarse_results, key=lambda r: r.get("sharpe", float("-inf")), reverse=True
        )
        k = max(1, int(len(sorted_res) * float(self.config.refine_top_pct)))
        top_params = [r.get("params", {}) for r in sorted_res[:k]]
        # Evaluate on full window/symbols with detailed model
        results: list[dict[str, Any]] = []
        # Optional progress bar for refine phase
        pbar = None
        if not getattr(self.config, "quiet_bars", False):
            try:
                from tqdm import tqdm  # type: ignore

                pbar = tqdm(total=len(top_params), desc="Refine", leave=True)
            except Exception:
                pbar = None
        for params in top_params:
            res = self._evaluate_params_with_prepared(params, prepared=None, phase1=False)
            results.append(res)
            if pbar is not None:
                try:
                    pbar.update(1)
                except Exception:
                    pass
        if pbar is not None:
            try:
                pbar.close()
            except Exception:
                pass
        return results

    def _evaluate_params_with_prepared(
        self, params: dict[str, Any], *, prepared=None, phase1: bool = False
    ) -> dict[str, Any]:
        """Evaluate a parameter set with optional prepared data and phase flags."""
        try:
            validated_params = self._validate_parameters(params)
            strategy = self.strategy_factory[self.config.parameter_space.strategy.name](
                validated_params
            )
            rules = self._create_portfolio_rules(validated_params)
            # Vectorized phase-1 fast path
            if phase1 and self.config.vectorized_phase1 and prepared is not None:
                return self._evaluate_vectorized_phase1(prepared, strategy, validated_params)
            # Run backtest
            from bot.backtest.engine_portfolio import run_backtest

            kwargs = dict(
                symbol=None,
                symbol_list_csv=None,
                start=DateValidator.validate_date(self.config.start_date),
                end=DateValidator.validate_date(self.config.end_date),
                strategy=strategy,
                rules=rules,
                regime_on=False if phase1 else True,
                exit_mode="signal" if phase1 else "stop",
                cooldown=int(validated_params.get("cooldown", 0)),
                entry_confirm=(
                    self.config.entry_confirm_phase1
                    if phase1 and self.config.vectorized_phase1
                    else int(validated_params.get("entry_confirm", 1))
                ),
                min_rebalance_pct=(
                    self.config.min_rebalance_pct_phase1
                    if phase1 and self.config.vectorized_phase1
                    else 0.0
                ),
                strict_mode=True,
                show_progress=False,
                make_plot=False,
                write_portfolio_csv=False,
                write_trades_csv=False,
                write_summary_csv=False,
                quiet_mode=True,
                prepared=prepared,
                return_summary=True,
            )
            # If prepared passed, run once using the prepared's symbols; otherwise iterate symbols list
            if prepared is not None:
                out = run_backtest(**kwargs)
                if out and isinstance(out, dict) and "summary" in out:
                    summary = out["summary"]
                    metrics = {
                        "params": validated_params,
                        "sharpe": float(summary.get("sharpe", 0.0)),
                        "cagr": float(summary.get("cagr", 0.0)),
                        "max_drawdown": float(summary.get("max_drawdown", 0.0)),
                        "n_trades": int(summary.get("n_trades", 0)),
                        "n_symbols": len(prepared.symbols),
                        "timestamp": datetime.now().isoformat(),
                    }
                    return metrics
                return {
                    "params": validated_params,
                    "sharpe": float("-inf"),
                    "cagr": float("-inf"),
                    "max_drawdown": float("inf"),
                    "n_trades": 0,
                }
            else:
                # No prepared: rely on run_backtest per symbol list in config via prepared=None
                out = run_backtest(**kwargs)
                if out and isinstance(out, dict) and "summary" in out:
                    summary = out["summary"]
                    metrics = {
                        "params": validated_params,
                        "sharpe": float(summary.get("sharpe", 0.0)),
                        "cagr": float(summary.get("cagr", 0.0)),
                        "max_drawdown": float(summary.get("max_drawdown", 0.0)),
                        "n_trades": int(summary.get("n_trades", 0)),
                        "n_symbols": len(self.config.symbols),
                        "timestamp": datetime.now().isoformat(),
                    }
                    return metrics
                return {
                    "params": validated_params,
                    "sharpe": float("-inf"),
                    "cagr": float("-inf"),
                    "max_drawdown": float("inf"),
                    "n_trades": 0,
                }
        except Exception as e:
            logger.error(f"Coarse/refine eval failed for {params}: {e}")
            return {
                "params": params,
                "sharpe": float("-inf"),
                "cagr": float("-inf"),
                "max_drawdown": float("inf"),
                "n_trades": 0,
            }

    def _evaluate_vectorized_phase1(
        self, prepared, strategy, params: dict[str, Any]
    ) -> dict[str, Any]:
        """Approximate, vectorized close-to-close model for coarse ranking (no ledger/costs)."""
        from bot.metrics.report import perf_metrics

        dates = prepared.dates_idx
        equity_curves: list[pd.Series] = []
        for sym in prepared.symbols:
            df = prepared.data_map.get(sym)
            if df is None or df.empty:
                continue
            # Signals (cache by (symbol, donchian, atr_period))
            key = (sym, int(params.get("donchian_lookback", 55)), int(params.get("atr_period", 20)))
            sig_df = self._signal_cache.get(key)
            if sig_df is None:
                sig_df = strategy.generate_signals(df.copy())
                self._signal_cache[key] = sig_df
            r = df.join(sig_df, how="left").reindex(dates)
            close = r["Close"].astype(float)
            ret = close.pct_change().fillna(0.0)
            pos = (
                (r.get("signal", pd.Series(0, index=r.index)).shift(1) > 0)
                .astype(float)
                .fillna(0.0)
            )
            strat_ret = pos * ret
            equity = (1.0 + strat_ret).cumprod()
            equity_curves.append(pd.Series(equity.values, index=dates))
        if not equity_curves:
            return {
                "params": params,
                "sharpe": float("-inf"),
                "cagr": float("-inf"),
                "max_drawdown": float("inf"),
                "n_trades": 0,
            }
        # Equal-weight aggregate across symbols by average returns
        # Use ffill() instead of deprecated fillna(method="ffill")
        eq_mat = np.column_stack(
            [s.reindex(dates).ffill().fillna(1.0).values for s in equity_curves]
        )
        eq_avg = np.nanmean(eq_mat, axis=1)
        equity_series = pd.Series(eq_avg, index=dates, name="equity")
        metrics = perf_metrics(equity_series.dropna())
        return {
            "params": params,
            "sharpe": float(metrics.get("sharpe", 0.0)),
            "cagr": float(metrics.get("cagr", 0.0)),
            "max_drawdown": float(metrics.get("max_drawdown", 0.0)),
            "n_trades": int(metrics.get("num_trades", 0)) if "num_trades" in metrics else 0,
            "n_symbols": len(equity_curves),
            "timestamp": datetime.now().isoformat(),
        }

    def _run_grid_search_optimized(self) -> list[dict[str, Any]]:
        """Run grid search with performance optimizations."""
        combinations = self.config.parameter_space.get_grid_combinations()

        if not combinations:
            logger.warning("No parameter combinations defined for grid search")
            return []

        logger.info(f"Grid search will evaluate {len(combinations)} combinations")

        # Use parallel processing for grid search
        if self.max_workers > 1:
            return self._evaluate_parallel(combinations, "Grid Search")
        else:
            return self._evaluate_sequential(combinations, "Grid Search")

    def _run_evolutionary_optimized(self) -> list[dict[str, Any]]:
        """Run evolutionary optimization with performance optimizations."""
        # Use the existing evolutionary optimizer but with optimized evaluation
        return self.evolutionary_optimizer.optimize(self._evaluate_parameters_optimized)

    def _maybe_load_seeds(self) -> None:
        """Load seeds from latest or specified file/run dir."""
        if not (self.config.seed_latest or self.config.seed_from):
            return
        seed_path: Path | None = None
        if self.config.seed_from:
            p = Path(self.config.seed_from)
            seed_path = (p / "seeds.json") if p.is_dir() else p
        else:
            # find most recent seeds.json in output_dir
            candidates = sorted(
                Path(self.config.output_dir).glob("**/seeds.json"),
                key=lambda x: x.stat().st_mtime,
                reverse=True,
            )
            seed_path = candidates[0] if candidates else None
        if not seed_path or not seed_path.exists():
            logger.info("No seed file found to load")
            return
        with open(seed_path) as f:
            obj = json.load(f)
        seed_list = obj.get("global", [])
        if not isinstance(seed_list, list) or not seed_list:
            logger.info("Seed file empty or invalid format")
            return
        # Merge or replace into evaluation cache as warm starts (no-op here), but store for evolutionary init
        self._warm_start_seeds = seed_list
        logger.info(f"Loaded {len(seed_list)} seed parameter sets from {seed_path}")

    def _write_seeds(self, topk: int = 5) -> None:
        """Write top-k parameter sets to seeds.json in the run directory."""
        if not self.results:
            return
        sorted_res = sorted(
            self.results, key=lambda r: r.get("sharpe", float("-inf")), reverse=True
        )
        picks = []
        for r in sorted_res[:topk]:
            params = r.get("params", {})
            # flatten and only keep known strategy params
            picks.append(params)
        out_dir = self.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "seeds.json"
        with open(out_path, "w") as f:
            json.dump({"global": picks}, f, indent=2)
        logger.info(f"Wrote seeds.json with top {len(picks)} params to {out_path}")

    def _evaluate_parallel(
        self, parameter_combinations: list[dict[str, Any]], description: str
    ) -> list[dict[str, Any]]:
        """Evaluate parameter combinations in parallel."""
        logger.info(f"Starting parallel evaluation with {self.max_workers} workers")

        try:
            from .parallel_evaluator import evaluate_batch_standalone

            # Optional progress bar
            pbar = None
            if not getattr(self.config, "quiet_bars", False):
                try:
                    from tqdm import tqdm  # type: ignore

                    pbar = tqdm(total=len(parameter_combinations), desc=description, leave=True)
                except Exception:
                    pbar = None

            # Prepare config data for standalone evaluation
            config_data = {
                "symbols": self.config.symbols,
                "start_date": self.config.start_date,
                "end_date": self.config.end_date,
            }

            # Create batches for parallel processing
            batch_size = max(1, len(parameter_combinations) // (self.max_workers * 4))
            batches = [
                parameter_combinations[i : i + batch_size]
                for i in range(0, len(parameter_combinations), batch_size)
            ]

            logger.info(
                f"Created {len(batches)} batches of size {batch_size} for parallel processing"
            )

            results = []
            completed = 0
            total = len(parameter_combinations)

            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit batch evaluation tasks
                future_to_batch = {
                    executor.submit(evaluate_batch_standalone, batch, config_data): batch
                    for batch in batches
                }

                # Collect results as they complete
                for future in as_completed(future_to_batch):
                    try:
                        batch_results = future.result()
                        results.extend(batch_results)
                        completed += len(batch_results)
                        if pbar is not None:
                            try:
                                pbar.update(len(batch_results))
                            except Exception:
                                pass

                        # Log progress
                        if completed % max(1, min(100, total // 20)) == 0 or completed == total:
                            logger.info(
                                f"{description} progress: {completed}/{total} ({completed/total*100:.1f}%)"
                            )

                    except Exception as e:
                        batch = future_to_batch[future]
                        logger.error(f"Failed to evaluate batch: {e}")
                        # Add error results for the batch
                        for params in batch:
                            results.append(
                                {
                                    "params": params,
                                    "sharpe": float("-inf"),
                                    "cagr": float("-inf"),
                                    "max_drawdown": float("inf"),
                                    "n_trades": 0,
                                    "error": str(e),
                                }
                            )

            if pbar is not None:
                try:
                    pbar.close()
                except Exception:
                    pass
            return results

        except Exception as e:
            logger.warning(
                f"Parallel processing failed: {e}. Falling back to sequential processing."
            )
            return self._evaluate_sequential(parameter_combinations, description)

    def _evaluate_sequential(
        self, parameter_combinations: list[dict[str, Any]], description: str
    ) -> list[dict[str, Any]]:
        """Evaluate parameter combinations sequentially with optimizations."""
        logger.info("Starting sequential evaluation with optimizations")

        results = []
        total = len(parameter_combinations)
        # Optional progress bar
        pbar = None
        if not getattr(self.config, "quiet_bars", False):
            try:
                from tqdm import tqdm  # type: ignore

                pbar = tqdm(total=total, desc=description, leave=True)
            except Exception:
                pbar = None

        # Batch processing for better performance
        batch_size = 10
        batches = [parameter_combinations[i : i + batch_size] for i in range(0, total, batch_size)]

        for _batch_idx, batch in enumerate(batches):
            batch_results = []

            for params in batch:
                try:
                    result = self._evaluate_parameters_optimized(params)
                    batch_results.append(result)

                    # Early termination for very poor results
                    if result.get("sharpe", 0) < -2.0:
                        logger.debug(
                            f"Early termination for poor result: {result.get('sharpe', 0):.4f}"
                        )

                except Exception as e:
                    logger.error(f"Failed to evaluate {params}: {e}")
                    batch_results.append(
                        {
                            "params": params,
                            "sharpe": float("-inf"),
                            "cagr": float("-inf"),
                            "max_drawdown": float("inf"),
                            "n_trades": 0,
                            "error": str(e),
                        }
                    )

            results.extend(batch_results)

            # Log progress
            completed = len(results)
            if pbar is not None:
                try:
                    pbar.update(len(batch_results))
                except Exception:
                    pass
            if completed % max(1, min(100, total // 20)) == 0 or completed == total:
                logger.info(
                    f"{description} progress: {completed}/{total} ({completed/total*100:.1f}%)"
                )

        if pbar is not None:
            try:
                pbar.close()
            except Exception:
                pass
        return results

    def _evaluate_parameters_optimized(self, params: dict[str, Any]) -> dict[str, Any]:
        """Evaluate strategy parameters with major performance optimizations."""
        try:
            # Check cache first
            param_key = tuple(sorted(params.items()))
            if param_key in self._evaluation_cache:
                return self._evaluation_cache[param_key]

            # Validate parameters
            validated_params = self._validate_parameters(params)

            # Create strategy
            strategy = self.strategy_factory[self.config.parameter_space.strategy.name](
                validated_params
            )

            # Create portfolio rules
            rules = self._create_portfolio_rules(validated_params)

            # Run backtest with major optimizations
            results = self._run_rapid_backtest_optimized(strategy, rules)

            if not results:
                result = {
                    "sharpe": float("-inf"),
                    "cagr": float("-inf"),
                    "max_drawdown": float("inf"),
                    "n_trades": 0,
                    "error": "No results",
                }
            else:
                # Aggregate results across symbols
                metrics = self._aggregate_results(results)

                # Add parameter info
                metrics["params"] = validated_params
                metrics["n_symbols"] = len(results)
                metrics["timestamp"] = datetime.now().isoformat()

                result = metrics

            # Cache the result
            self._evaluation_cache[param_key] = result

            # Track diverse strategies
            self.diversity_tracker.add_strategy(validated_params, result)

            return result

        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return {
                "sharpe": float("-inf"),
                "cagr": float("-inf"),
                "max_drawdown": float("inf"),
                "n_trades": 0,
                "error": str(e),
            }

    def _run_rapid_backtest_optimized(self, strategy, rules) -> list[dict[str, Any]]:
        """Run backtest with major performance optimizations."""
        from bot.backtest.engine_portfolio import run_backtest

        all_results = []

        # Use pre-loaded data if available
        if self.market_data:
            # Run optimized backtest with pre-loaded data
            for symbol, data in self.market_data.items():
                try:
                    result = self._run_backtest_with_data(strategy, rules, symbol, data)
                    if result:
                        all_results.append(result)
                except Exception as e:
                    logger.warning(f"Failed to backtest {symbol}: {e}")
        else:
            # Fallback to regular backtest
            for symbol in self.config.symbols:
                try:
                    result = run_backtest(
                        symbol=symbol,
                        symbol_list_csv=None,
                        start=DateValidator.validate_date(self.config.start_date),
                        end=DateValidator.validate_date(self.config.end_date),
                        strategy=strategy,
                        rules=rules,
                        write_trades_csv=False,
                        write_summary_csv=False,
                        write_portfolio_csv=False,
                        quiet_mode=True,
                        return_summary=True,
                        show_progress=False,
                        make_plot=False,
                        prepared=None,
                    )
                    if result:
                        all_results.append(result)
                except Exception as e:
                    logger.warning(f"Failed to backtest {symbol}: {e}")

        return all_results

    def _run_backtest_with_data(
        self, strategy, rules, symbol: str, data: pd.DataFrame
    ) -> dict[str, Any] | None:
        """Run backtest using pre-loaded data for maximum speed."""
        try:
            from bot.exec.ledger import Ledger
            from bot.metrics.report import perf_metrics
            from bot.portfolio.allocator import allocate_signals

            # Create ledger
            ledger = Ledger()

            # Generate signals using pre-loaded data
            signals = strategy.generate_signals(data)

            # Allocate positions
            positions = allocate_signals(signals, rules, data)

            # Execute trades
            for date, position in positions.items():
                if position != 0:
                    ledger.record_trade(symbol, date, position, data.loc[date, "Close"])

            # Calculate performance metrics
            if ledger.trades:
                equity_curve = ledger.get_equity_curve()
                metrics = perf_metrics(equity_curve)

                return {
                    "sharpe": metrics.get("sharpe", 0),
                    "cagr": metrics.get("cagr", 0),
                    "max_drawdown": metrics.get("max_drawdown", 0),
                    "n_trades": len(ledger.trades),
                }

            return None

        except Exception as e:
            logger.warning(f"Failed to run optimized backtest for {symbol}: {e}")
            return None

    def _evaluate_parameters(self, params: dict[str, Any]) -> dict[str, Any]:
        """Legacy evaluation function for compatibility."""
        return self._evaluate_parameters_optimized(params)

    def _aggregate_results(self, results: list[dict[str, Any]]) -> dict[str, Any]:
        """Aggregate results across symbols with rapid evolution improvements."""
        if not results:
            return {
                "sharpe": float("-inf"),
                "cagr": float("-inf"),
                "max_drawdown": float("inf"),
                "n_trades": 0,
            }

        # Calculate weighted averages based on performance
        sharpes = [r.get("sharpe", 0) for r in results]
        cagrs = [r.get("cagr", 0) for r in results]
        drawdowns = [r.get("max_drawdown", 0) for r in results]
        trades = [r.get("n_trades", 0) for r in results]

        # Use simple averages for speed
        avg_sharpe = np.mean(sharpes)
        avg_cagr = np.mean(cagrs)
        avg_max_dd = np.mean(drawdowns)
        total_trades = sum(trades)

        return {
            "sharpe": avg_sharpe,
            "cagr": avg_cagr,
            "max_drawdown": avg_max_dd,
            "n_trades": total_trades,
        }

    def _validate_parameters(self, params: dict[str, Any]) -> dict[str, Any]:
        """Validate and normalize parameters."""
        validated = {}
        strategy_config = self.config.parameter_space.strategy

        for param_name, param_def in strategy_config.parameters.items():
            if param_name in params:
                value = params[param_name]

                # Type validation
                if param_def.type == "int":
                    validated[param_name] = int(value)
                elif param_def.type == "float":
                    validated[param_name] = float(value)
                elif param_def.type == "bool":
                    validated[param_name] = bool(value)
                else:
                    validated[param_name] = value
            else:
                validated[param_name] = param_def.default

        return validated

    def _create_trend_breakout_strategy(self, params: dict[str, Any]):
        """Create trend breakout strategy."""
        from bot.strategy.trend_breakout import TrendBreakoutParams, TrendBreakoutStrategy

        strategy_params = TrendBreakoutParams(
            donchian_lookback=params.get("donchian_lookback", 55),
            atr_period=params.get("atr_period", 20),
            atr_k=params.get("atr_k", 2.0),
        )
        return TrendBreakoutStrategy(strategy_params)

    def _create_demo_ma_strategy(self, params: dict[str, Any]):
        """Create demo MA strategy."""
        from bot.strategy.demo_ma import DemoMAParams, DemoMAStrategy

        strategy_params = DemoMAParams(
            short_window=params.get("short_window", 10),
            long_window=params.get("long_window", 50),
        )
        return DemoMAStrategy(strategy_params)

    def _create_portfolio_rules(self, params: dict[str, Any]):
        """Create portfolio rules."""
        from bot.portfolio.allocator import PortfolioRules

        return PortfolioRules(
            per_trade_risk_pct=params.get("risk_pct", 0.5) / 100.0,
            atr_k=params.get("atr_k", 2.0),
            max_positions=10,
            cost_bps=5.0,
        )

    def _generate_analysis(self) -> None:
        """Generate analysis with rapid evolution improvements."""
        if not self.results:
            logger.warning("No results to analyze")
            return

        # Create analysis directory
        analysis_dir = self.output_dir / "analysis"
        analysis_dir.mkdir(exist_ok=True)

        # Generate diversity analysis
        self.diversity_tracker.save_diverse_strategies()
        self.diversity_tracker.cluster_strategies(n_clusters=5)

        # Generate analysis report
        analysis = self.analyzer.analyze_results(self.results)
        analysis = _convert_numpy_types(analysis)  # Convert numpy types

        # Add diversity summary to main analysis
        diversity_summary = {
            "diverse_strategies_count": len(self.diversity_tracker.diverse_strategies),
            "strategy_clusters": len(self.diversity_tracker.strategy_clusters),
            "recommendations": self.diversity_tracker.get_strategy_recommendations(),
        }
        analysis["diversity_analysis"] = diversity_summary

        # Save analysis report
        analysis_path = analysis_dir / "analysis.json"
        with open(analysis_path, "w") as f:
            json.dump(analysis, f, indent=2, default=str)

        # Generate visualizations (only if requested)
        if self.config.create_plots:
            try:
                self.visualizer.create_dashboard(self.results, analysis_dir)
                logger.info("Visualizations created successfully")
            except Exception as e:
                logger.warning(f"Failed to create visualizations: {e}")

    def _generate_summary(self) -> dict[str, Any]:
        """Generate optimization summary with rapid evolution improvements."""
        if not self.results:
            return {
                "config": {
                    "name": self.config.name,
                    "description": self.config.description,
                    "strategy": getattr(self.config.parameter_space.strategy, "name", "unknown"),
                    "symbols": self.config.symbols,
                    "date_range": f"{self.config.start_date} to {self.config.end_date}",
                },
                "total_evaluations": 0,
                "best_result": None,
                "statistics": {},
                "strategy_types": {},
                "diversity_analysis": {
                    "diverse_strategies_found": 0,
                    "strategy_clusters": 0,
                },
                "timestamp": datetime.now().isoformat(),
                "error": "No results found",
            }

        # Calculate statistics
        sharpes = [r.get("sharpe", 0) for r in self.results]
        cagrs = [r.get("cagr", 0) for r in self.results]
        drawdowns = [r.get("max_drawdown", 0) for r in self.results]
        trades = [r.get("n_trades", 0) for r in self.results]

        # Identify strategy types
        strategy_types = self._identify_strategy_types()

        summary = {
            "config": {
                "name": self.config.name,
                "description": self.config.description,
                "strategy": self.config.parameter_space.strategy.name,
                "symbols": self.config.symbols,
                "date_range": f"{self.config.start_date} to {self.config.end_date}",
            },
            "total_evaluations": len(self.results),
            "best_result": self.best_result,
            "statistics": {
                "sharpe": {
                    "mean": np.mean(sharpes),
                    "max": max(sharpes),
                    "min": min(sharpes),
                    "std": np.std(sharpes),
                },
                "cagr": {
                    "mean": np.mean(cagrs),
                    "max": max(cagrs),
                    "min": min(cagrs),
                },
                "max_drawdown": {
                    "mean": np.mean(drawdowns),
                    "min": min(drawdowns),
                    "max": max(drawdowns),
                },
                "n_trades": {
                    "mean": np.mean(trades),
                    "total": sum(trades),
                },
            },
            "strategy_types": strategy_types,
            "diversity_analysis": {
                "diverse_strategies_found": len(self.diversity_tracker.diverse_strategies),
                "strategy_clusters": len(self.diversity_tracker.strategy_clusters),
            },
            "timestamp": datetime.now().isoformat(),
        }

        return summary

    def _identify_strategy_types(self) -> dict[str, int]:
        """Identify different types of strategies."""
        types = {
            "conservative": 0,
            "aggressive": 0,
            "short_term": 0,
            "long_term": 0,
            "high_frequency": 0,
        }

        for result in self.results:
            params = result.get("params", {})

            # Conservative (low risk, high confirmation)
            if params.get("risk_pct", 0) < 0.5 and params.get("entry_confirm", 0) > 2:
                types["conservative"] += 1

            # Aggressive (high risk, low confirmation)
            if params.get("risk_pct", 0) > 2.0 and params.get("entry_confirm", 0) <= 1:
                types["aggressive"] += 1

            # Short-term (short lookback periods)
            if params.get("donchian_lookback", 0) < 50:
                types["short_term"] += 1

            # Long-term (long lookback periods)
            if params.get("donchian_lookback", 0) > 200:
                types["long_term"] += 1

            # High-frequency (no cooldown, no confirmation)
            if params.get("cooldown", 0) == 0 and params.get("entry_confirm", 0) == 0:
                types["high_frequency"] += 1

        return types
