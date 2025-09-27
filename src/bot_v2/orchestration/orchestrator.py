"""
Core Trading Orchestrator for Bot V2 (legacy equities workflows).

This orchestrator connects feature slices (data/analyze/backtest/etc.) into a unified
system and implements graceful degradation when slices are unavailable.

Important: The primary production path is the Coinbase Perpetual Futures bot
(`bot_v2.orchestration.perps_bot`). This module remains for legacy workflows and tests
and will be incrementally deprecated.
"""
import logging
from typing import Any, Optional
from datetime import datetime, timedelta
from dataclasses import asdict

from .types import TradingMode, OrchestratorConfig, OrchestrationResult
from ..data_providers import get_data_provider

logger = logging.getLogger(__name__)


class TradingOrchestrator:
    """
    Core orchestration engine that connects all feature slices.
    Implements graceful degradation when slices are unavailable.
    """
    
    def __init__(self, config: OrchestratorConfig | None = None):
        """Initialize orchestrator with configuration and load slices"""
        self.config = config or OrchestratorConfig()
        self.logger = logging.getLogger("orchestrator")
        
        # Track slice availability
        self.available_slices = {}
        self.failed_slices = {}
        
        # Initialize all slices with error handling
        self._initialize_slices()
        
        # Report status
        self.logger.info(f"Orchestrator initialized: {len(self.available_slices)}/11 slices available")
        if self.failed_slices:
            self.logger.warning(f"Failed slices: {list(self.failed_slices.keys())}")
    
    def _initialize_slices(self):
        """Initialize all feature slices with graceful error handling"""
        
        # Data slice
        try:
            from ..features.data import data
            self.data = data
            self.available_slices['data'] = True
        except ImportError as e:
            self.logger.warning(f"Could not import data slice: {e}")
            self.data = None
            self.failed_slices['data'] = str(e)
        
        # Analyze slice
        try:
            from ..features.analyze import analyze
            self.analyzer = analyze
            self.available_slices['analyze'] = True
        except ImportError as e:
            self.logger.warning(f"Could not import analyze slice: {e}")
            self.analyzer = None
            self.failed_slices['analyze'] = str(e)
        
        # Market regime slice
        try:
            from ..features.market_regime import market_regime
            self.market_regime = market_regime
            self.available_slices['market_regime'] = True
        except ImportError as e:
            self.logger.warning(f"Could not import market_regime slice: {e}")
            self.market_regime = None
            self.failed_slices['market_regime'] = str(e)
        
        # ML strategy slice
        try:
            from ..features.ml_strategy import ml_strategy
            self.ml_strategy = ml_strategy
            self.available_slices['ml_strategy'] = True
        except ImportError as e:
            self.logger.warning(f"Could not import ml_strategy slice: {e}")
            self.ml_strategy = None
            self.failed_slices['ml_strategy'] = str(e)
        
        # Position sizing slice
        try:
            from ..features.position_sizing import position_sizing
            self.position_sizing = position_sizing
            self.available_slices['position_sizing'] = True
        except ImportError as e:
            self.logger.warning(f"Could not import position_sizing slice: {e}")
            self.position_sizing = None
            self.failed_slices['position_sizing'] = str(e)
        
        # Backtest slice
        try:
            from ..features.backtest import backtest
            self.backtest = backtest
            self.available_slices['backtest'] = True
        except ImportError as e:
            self.logger.warning(f"Could not import backtest slice: {e}")
            self.backtest = None
            self.failed_slices['backtest'] = str(e)
        
        # Paper trade slice
        try:
            from ..features.paper_trade import paper_trade
            self.paper_trade = paper_trade
            self.available_slices['paper_trade'] = True
        except ImportError as e:
            self.logger.warning(f"Could not import paper_trade slice: {e}")
            self.paper_trade = None
            self.failed_slices['paper_trade'] = str(e)
        
        # Live trade slice
        try:
            from ..features.live_trade import live_trade
            self.live_trade = live_trade
            self.available_slices['live_trade'] = True
        except ImportError as e:
            self.logger.warning(f"Could not import live_trade slice: {e}")
            self.live_trade = None
            self.failed_slices['live_trade'] = str(e)
        
        # Monitor slice
        try:
            from ..features.monitor import monitor
            self.monitor = monitor
            self.available_slices['monitor'] = True
        except ImportError as e:
            self.logger.warning(f"Could not import monitor slice: {e}")
            self.monitor = None
            self.failed_slices['monitor'] = str(e)
        
        # Optimize slice
        try:
            from ..features.optimize import optimize
            self.optimizer = optimize
            self.available_slices['optimize'] = True
        except ImportError as e:
            self.logger.warning(f"Could not import optimize slice: {e}")
            self.optimizer = None
            self.failed_slices['optimize'] = str(e)
        
        # Adaptive portfolio slice
        try:
            from ..features.adaptive_portfolio import adaptive_portfolio
            self.adaptive_portfolio = adaptive_portfolio
            self.available_slices['adaptive_portfolio'] = True
        except ImportError as e:
            self.logger.warning(f"Could not import adaptive_portfolio slice: {e}")
            self.adaptive_portfolio = None
            self.failed_slices['adaptive_portfolio'] = str(e)

    def _resolve_optimize_strategy_name(self, candidate: str | None) -> str:
        """Normalize strategy identifiers for the optimize slice."""
        if not candidate:
            return "Momentum"

        normalized = candidate.replace(" ", "").replace("_", "").lower()
        strategy_map = {
            'momentum': 'Momentum',
            'momentumstrategy': 'Momentum',
            'simplema': 'SimpleMA',
            'macrossover': 'SimpleMA',
            'meanreversion': 'MeanReversion',
            'meanreversionstrategy': 'MeanReversion',
            'volatility': 'Volatility',
            'volatilitystrategy': 'Volatility',
            'breakout': 'Breakout',
            'breakoutstrategy': 'Breakout',
        }
        return strategy_map.get(normalized, candidate)
    
    def execute_trading_cycle(self, symbol: str) -> OrchestrationResult:
        """
        Execute complete trading cycle for a symbol.
        Flow: data → analyze → market_regime → ml_strategy → position_sizing → execute → monitor
        """
        self.logger.info(f"Starting trading cycle for {symbol}")
        
        errors = []
        data_result = {}
        metrics = {}
        start_time = datetime.now()
        
        try:
            # Step 1: Fetch data using data provider abstraction
            market_data = None
            try:
                provider = get_data_provider()
                market_data = provider.get_historical_data(symbol, period="60d")
                
                if market_data is not None and not market_data.empty:
                    data_result['data_fetched'] = True
                    metrics['data_rows'] = len(market_data)
                    self.logger.info(f"Fetched {len(market_data)} rows of data for {symbol}")
                else:
                    errors.append(f"No data returned for {symbol}")
                    data_result['data_fetched'] = False
            except Exception as e:
                errors.append(f"Data fetch failed: {e}")
                data_result['data_fetched'] = False
            
            # Step 2: Analyze market
            analysis = {}
            if self.analyzer and hasattr(self.analyzer, 'analyze_symbol'):
                try:
                    # Use new API: analyze_symbol(symbol, lookback_days) instead of analyze(market_data, symbol)
                    analysis_result = self.analyzer.analyze_symbol(symbol, lookback_days=60)
                    analysis = {
                        'recommendation': analysis_result.recommendation,
                        'confidence': analysis_result.confidence,
                        'indicators': analysis_result.indicators,
                        'regime': analysis_result.regime
                    }
                    data_result['analysis'] = analysis
                except Exception as e:
                    errors.append(f"Analysis failed: {e}")
            
            # Step 3: Detect market regime
            regime = "unknown"
            if self.config.enable_regime_detection and self.market_regime:
                try:
                    if hasattr(self.market_regime, 'detect_regime'):
                        # Use new API: detect_regime(symbol, lookback_days) instead of detect_regime(market_data)
                        regime_analysis = self.market_regime.detect_regime(symbol, lookback_days=60)
                        regime = regime_analysis.current_regime.value if hasattr(regime_analysis.current_regime, 'value') else str(regime_analysis.current_regime)
                        data_result['regime'] = regime
                        data_result['regime_confidence'] = regime_analysis.confidence
                except Exception as e:
                    errors.append(f"Regime detection failed: {e}")
            
            # Step 4: Select strategy using ML
            strategy = "momentum"  # default
            confidence = 0.5
            if self.config.enable_ml_strategy and self.ml_strategy:
                try:
                    if hasattr(self.ml_strategy, 'predict_best_strategy'):
                        # Use new API: predict_best_strategy(symbol, lookback_days, top_n=1)
                        predictions = self.ml_strategy.predict_best_strategy(symbol, lookback_days=60, top_n=1)
                        if predictions and len(predictions) > 0:
                            best_prediction = predictions[0]
                            strategy = best_prediction.strategy.value if hasattr(best_prediction.strategy, 'value') else str(best_prediction.strategy)
                            confidence = best_prediction.confidence
                            data_result['strategy'] = strategy
                            data_result['confidence'] = confidence
                            data_result['expected_return'] = best_prediction.expected_return
                        else:
                            errors.append("ML strategy selection returned no predictions")
                except Exception as e:
                    errors.append(f"ML strategy selection failed: {e}")
            
            # Step 5: Calculate position size and convert to shares
            position_size = min(0.1 * confidence, 0.2)  # Scale by confidence, max 20%

            # Get current price and calculate shares
            capital_to_invest = position_size * self.config.capital
            current_price: float | None = None
            try:
                provider = get_data_provider()
                current_price = provider.get_current_price(symbol)
                if current_price <= 0:
                    raise ValueError("current price must be positive")
                shares = max(int(capital_to_invest / current_price), 1)  # Convert dollars to shares

                data_result['position_size'] = position_size
                data_result['shares'] = shares
                data_result['capital_invested'] = capital_to_invest
                data_result['current_price'] = current_price
                self.logger.info(
                    "Position size: %s, Shares: %s, Capital: $%s",
                    f"{position_size:.2%}",
                    shares,
                    f"{capital_to_invest:.2f}",
                )
            except Exception as e:
                errors.append(f"Position sizing failed: {e}")
                shares = 1  # Fallback to at least one share
            # Step 6: Execute based on mode
            execution_result = {}
            if self.config.mode == TradingMode.BACKTEST and self.backtest:
                try:
                    if hasattr(self.backtest, 'run_backtest'):
                        # Calculate date range for backtest
                        end_date = datetime.now()
                        start_date = end_date - timedelta(days=90)  # 3 months backtest
                        
                        execution_result = self.backtest.run_backtest(
                            strategy=strategy,
                            symbol=symbol,
                            start=start_date,
                            end=end_date,
                            initial_capital=self.config.capital
                        )
                        data_result['backtest_result'] = execution_result
                except Exception as e:
                    errors.append(f"Backtest execution failed: {e}")
            
            elif self.config.mode == TradingMode.OPTIMIZE:
                if self.optimizer and hasattr(self.optimizer, 'optimize_strategy'):
                    try:
                        optimize_fn = getattr(self.optimizer, 'optimize_strategy')
                        strategy_candidate = (
                            self.config.strategies[0]
                            if self.config.strategies else strategy
                        )
                        strategy_name = self._resolve_optimize_strategy_name(strategy_candidate)
                        strategy = strategy_name
                        data_result['strategy'] = strategy_name

                        # Use trailing year for optimization window
                        optimization_end = datetime.now()
                        optimization_start = optimization_end - timedelta(days=365)

                        optimization_result = optimize_fn(
                            strategy=strategy_name,
                            symbol=symbol,
                            start_date=optimization_start,
                            end_date=optimization_end,
                        )

                        period = getattr(optimization_result, 'period', None)
                        best_metrics_obj = getattr(optimization_result, 'best_metrics', None)
                        combinations_tested = len(getattr(
                            optimization_result,
                            'all_results',
                            []
                        ) or [])

                        try:
                            best_metrics = asdict(best_metrics_obj) if best_metrics_obj else {}
                        except TypeError:
                            best_metrics = dict(getattr(best_metrics_obj, '__dict__', {}) or {})

                        optimization_payload = {
                            'strategy': strategy_name,
                            'period': {
                                'start': (period[0] if period else optimization_start).isoformat(),
                                'end': (period[1] if period else optimization_end).isoformat(),
                            },
                            'best_params': getattr(optimization_result, 'best_params', {}),
                            'best_metrics': best_metrics,
                            'combinations_tested': combinations_tested,
                        }
                        if hasattr(optimization_result, 'summary'):
                            try:
                                optimization_payload['summary'] = optimization_result.summary()
                            except Exception:
                                pass

                        data_result['optimization'] = optimization_payload
                        metrics['optimize_time'] = float(
                            getattr(optimization_result, 'optimization_time', 0.0) or 0.0
                        )
                        metrics['optimize_combinations'] = combinations_tested
                        execution_result = optimization_result
                        self.logger.info(
                            "Optimization completed for %s using %s (%s combinations)",
                            symbol,
                            strategy_name,
                            combinations_tested,
                        )
                    except Exception as e:
                        errors.append(f"Optimization failed: {e}")
                else:
                    errors.append("Optimization slice not available")

            elif self.config.mode == TradingMode.PAPER and self.paper_trade:
                try:
                    if hasattr(self.paper_trade, 'execute_paper_trade'):
                        execution_result = self.paper_trade.execute_paper_trade(
                            symbol=symbol,
                            action='buy',
                            quantity=shares,
                            strategy_info={
                                'strategy': strategy,
                                'confidence': confidence,
                                'capital_allocated': capital_to_invest,
                                'reference_price': current_price,
                            },
                        )
                        data_result['paper_trade_result'] = execution_result
                except Exception as e:
                    errors.append(f"Paper trade execution failed: {e}")
            
            elif self.config.mode == TradingMode.LIVE and self.live_trade:
                try:
                    if hasattr(self.live_trade, 'execute_live_trade'):
                        execution_result = self.live_trade.execute_live_trade(
                            symbol=symbol,
                            action='buy',
                            quantity=shares,
                            strategy_info={
                                'strategy': strategy,
                                'confidence': confidence,
                                'capital_allocated': capital_to_invest,
                                'reference_price': current_price,
                            },
                        )
                        data_result['live_trade_result'] = execution_result
                except Exception as e:
                    errors.append(f"Live trade execution failed: {e}")
            
            # Step 7: Monitor and log
            if self.monitor:
                try:
                    # Import log_event directly from monitor.logger
                    from ..features.monitor.logger import log_event
                    log_event(
                        f"Trading cycle completed for {symbol}",
                        {
                            'type': 'trading_cycle',
                            'symbol': symbol,
                            'mode': self.config.mode.value,
                            'strategy': strategy,
                            'position_size': position_size,
                            'shares': shares,
                            'errors': len(errors)
                        }
                    )
                except Exception as e:
                    errors.append(f"Monitoring failed: {e}")
            
            # Calculate metrics
            end_time = datetime.now()
            metrics['execution_time'] = (end_time - start_time).total_seconds()
            metrics['errors_count'] = len(errors)
            metrics['slices_used'] = len([k for k, v in self.available_slices.items() if v])
            
            # Determine success
            success = len(errors) == 0 or (len(errors) < 3 and data_result.get('data_fetched', False))
            
            return OrchestrationResult(
                timestamp=end_time,
                mode=self.config.mode,
                symbol=symbol,
                success=success,
                data=data_result,
                errors=errors,
                metrics=metrics
            )
            
        except Exception as e:
            self.logger.error(f"Unexpected error in trading cycle: {e}")
            errors.append(f"Unexpected error: {e}")
            
            return OrchestrationResult(
                timestamp=datetime.now(),
                mode=self.config.mode,
                symbol=symbol,
                success=False,
                data=data_result,
                errors=errors,
                metrics=metrics
            )
    
    def get_status(self) -> dict[str, Any]:
        """Get orchestrator status"""
        return {
            'available_slices': self.available_slices,
            'failed_slices': self.failed_slices,
            'slice_availability': f"{len(self.available_slices)}/11",
            'mode': self.config.mode.value,
            'symbols': self.config.symbols,
            'capital': self.config.capital
        }
    
    def run(self, symbols: list[str | None] = None) -> list[OrchestrationResult]:
        """Run trading cycles for multiple symbols"""
        symbols = symbols or self.config.symbols
        results = []
        
        for symbol in symbols:
            try:
                result = self.execute_trading_cycle(symbol)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Failed to process {symbol}: {e}")
                results.append(OrchestrationResult(
                    timestamp=datetime.now(),
                    mode=self.config.mode,
                    symbol=symbol,
                    success=False,
                    data={},
                    errors=[str(e)],
                    metrics={}
                ))
        
        return results
