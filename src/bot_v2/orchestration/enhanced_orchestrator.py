"""
Enhanced Trading Orchestrator with Full ML Integration (legacy equities workflows).

This orchestrator extends the base orchestrator with complete ML pipeline integration,
including caching, monitoring, and advanced decision making.

Note: The active production path for Coinbase Perpetual Futures uses `perps_bot.py`.
This module remains for legacy workflows/tests and is slated for deprecation.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

from .orchestrator import TradingOrchestrator
from .ml_integration import MLPipelineIntegrator, MLDecision
from .types import TradingMode, OrchestratorConfig, OrchestrationResult


class EnhancedTradingOrchestrator(TradingOrchestrator):
    """
    Enhanced orchestrator with full ML pipeline integration.
    Extends base orchestrator with ML-driven decision making.
    """
    
    def __init__(self, config: Optional[OrchestratorConfig] = None):
        """Initialize enhanced orchestrator with ML integration"""
        super().__init__(config)
        
        # Initialize ML integrator
        ml_config = {
            'min_confidence': config.min_confidence if config else 0.6,
            'max_position_size': config.max_position_pct if config else 0.2,
            'enable_caching': True,
            'cache_ttl_minutes': 5
        }
        
        self.ml_integrator = MLPipelineIntegrator(ml_config)
        self.logger.info("Enhanced orchestrator with ML integration initialized")
        
        # Track ML decisions for monitoring
        self.ml_decisions_history = []
        self.ml_performance_metrics = {
            'total_decisions': 0,
            'buy_signals': 0,
            'sell_signals': 0,
            'hold_signals': 0,
            'avg_confidence': 0.0,
            'successful_trades': 0
        }
    
    def execute_ml_trading_cycle(
        self, 
        symbols: List[str],
        current_positions: Optional[Dict[str, float]] = None
    ) -> Dict[str, OrchestrationResult]:
        """
        Execute ML-driven trading cycle for multiple symbols.
        
        Args:
            symbols: List of symbols to trade
            current_positions: Current portfolio positions
            
        Returns:
            Dictionary of orchestration results by symbol
        """
        self.logger.info(f"Starting ML trading cycle for {len(symbols)} symbols")
        
        results = {}
        portfolio_value = self.config.capital
        
        # Get ML decisions for all symbols
        ml_decisions = self.ml_integrator.get_portfolio_ml_decisions(
            symbols=symbols,
            portfolio_value=portfolio_value,
            current_positions=current_positions or {}
        )
        
        # Process each decision
        for decision in ml_decisions:
            try:
                result = self._execute_ml_decision(decision, portfolio_value)
                results[decision.symbol] = result
                
                # Update metrics
                self._update_ml_metrics(decision)
                
            except Exception as e:
                self.logger.error(f"Failed to execute ML decision for {decision.symbol}: {e}")
                results[decision.symbol] = self._create_error_result(str(e), decision.symbol)
        
        # Log ML performance
        self._log_ml_performance()
        
        return results
    
    def _execute_ml_decision(
        self, 
        decision: MLDecision,
        portfolio_value: float
    ) -> OrchestrationResult:
        """Execute a single ML decision"""
        
        self.logger.info(
            f"Executing ML decision for {decision.symbol}: "
            f"{decision.decision} using {decision.strategy} "
            f"(confidence: {decision.confidence:.1%})"
        )
        
        # Store decision in history
        self.ml_decisions_history.append(decision)
        
        # Build execution context
        execution_context = {
            'symbol': decision.symbol,
            'strategy': decision.strategy,
            'action': decision.decision,
            'confidence': decision.confidence,
            'expected_return': decision.expected_return,
            'regime': decision.regime,
            'position_size': decision.risk_adjusted_size,
            'reasoning': decision.reasoning
        }
        
        # Calculate shares to trade
        if decision.decision != 'hold':
            try:
                from ..data_providers import get_data_provider
                provider = get_data_provider()
                current_price = provider.get_current_price(decision.symbol)
                
                capital_to_invest = decision.risk_adjusted_size * portfolio_value
                shares = int(capital_to_invest / current_price)
                
                execution_context['shares'] = shares
                execution_context['capital'] = capital_to_invest
                execution_context['price'] = current_price
                
            except Exception as e:
                self.logger.warning(f"Failed to calculate shares: {e}")
                execution_context['shares'] = 0
        
        # Execute based on mode and action
        if decision.decision == 'hold':
            return OrchestrationResult(
                timestamp=datetime.now(),
                symbol=decision.symbol,
                success=True,
                mode=self.config.mode,
                data=execution_context,
                errors=[],
                execution_time=(datetime.now() - decision.timestamp).total_seconds(),
                metrics={'action': 'hold', 'confidence': decision.confidence}
            )
        
        # Execute trade based on mode
        if self.config.mode == TradingMode.BACKTEST:
            return self._execute_backtest_ml(execution_context)
        elif self.config.mode == TradingMode.PAPER:
            return self._execute_paper_trade_ml(execution_context)
        elif self.config.mode == TradingMode.LIVE:
            return self._execute_live_trade_ml(execution_context)
        else:
            return self._create_error_result(f"Unknown mode: {self.config.mode}", decision.symbol)
    
    def _execute_backtest_ml(self, context: Dict) -> OrchestrationResult:
        """Execute ML decision in backtest mode"""
        if not self.backtest:
            return self._create_error_result("Backtest slice not available", context['symbol'])
        
        try:
            # Run backtest with ML-selected strategy
            end_date = datetime.now()
            start_date = end_date - timedelta(days=90)
            
            result = self.backtest.run_backtest(
                strategy=context['strategy'],
                symbol=context['symbol'],
                start=start_date,
                end=end_date,
                initial_capital=self.config.capital
            )
            
            return OrchestrationResult(
                timestamp=datetime.now(),
                symbol=context['symbol'],
                success=True,
                mode=TradingMode.BACKTEST,
                data={'backtest_result': result, **context},
                errors=[],
                execution_time=0.0,
                metrics=result.metrics if hasattr(result, 'metrics') else {}
            )
            
        except Exception as e:
            return self._create_error_result(f"Backtest failed: {e}", context['symbol'])
    
    def _execute_paper_trade_ml(self, context: Dict) -> OrchestrationResult:
        """Execute ML decision in paper trading mode"""
        if not self.paper_trade:
            return self._create_error_result("Paper trade slice not available", context['symbol'])
        
        try:
            # Execute paper trade with ML decision
            result = self.paper_trade.execute_paper_trade(
                symbol=context['symbol'],
                action=context['action'],
                quantity=context.get('shares', 1),
                strategy_info={
                    'strategy': context['strategy'],
                    'confidence': context['confidence'],
                    'regime': context['regime']
                }
            )
            
            return OrchestrationResult(
                timestamp=datetime.now(),
                symbol=context['symbol'],
                success=True,
                mode=TradingMode.PAPER,
                data={'paper_trade_result': result, **context},
                errors=[],
                execution_time=0.0,
                metrics={'executed': True}
            )
            
        except Exception as e:
            return self._create_error_result(f"Paper trade failed: {e}", context['symbol'])
    
    def _execute_live_trade_ml(self, context: Dict) -> OrchestrationResult:
        """Execute ML decision in live trading mode"""
        if not self.live_trade:
            return self._create_error_result("Live trade slice not available", context['symbol'])
        
        try:
            # Add safety check for live trading
            if context['confidence'] < 0.7:
                return self._create_error_result(
                    f"Confidence {context['confidence']:.1%} too low for live trading",
                    context['symbol']
                )
            
            # Execute live trade with ML decision
            result = self.live_trade.execute_live_trade(
                symbol=context['symbol'],
                action=context['action'],
                quantity=context.get('shares', 1),
                strategy_info={
                    'strategy': context['strategy'],
                    'confidence': context['confidence'],
                    'regime': context['regime']
                }
            )
            
            return OrchestrationResult(
                timestamp=datetime.now(),
                symbol=context['symbol'],
                success=True,
                mode=TradingMode.LIVE,
                data={'live_trade_result': result, **context},
                errors=[],
                execution_time=0.0,
                metrics={'executed': True}
            )
            
        except Exception as e:
            return self._create_error_result(f"Live trade failed: {e}", context['symbol'])
    
    def _update_ml_metrics(self, decision: MLDecision):
        """Update ML performance metrics"""
        self.ml_performance_metrics['total_decisions'] += 1
        
        if decision.decision == 'buy':
            self.ml_performance_metrics['buy_signals'] += 1
        elif decision.decision == 'sell':
            self.ml_performance_metrics['sell_signals'] += 1
        else:
            self.ml_performance_metrics['hold_signals'] += 1
        
        # Update average confidence
        n = self.ml_performance_metrics['total_decisions']
        prev_avg = self.ml_performance_metrics['avg_confidence']
        self.ml_performance_metrics['avg_confidence'] = (
            (prev_avg * (n - 1) + decision.confidence) / n
        )
    
    def _log_ml_performance(self):
        """Log ML performance metrics"""
        metrics = self.ml_performance_metrics
        self.logger.info(
            f"ML Performance - Total: {metrics['total_decisions']}, "
            f"Buy: {metrics['buy_signals']}, "
            f"Sell: {metrics['sell_signals']}, "
            f"Hold: {metrics['hold_signals']}, "
            f"Avg Confidence: {metrics['avg_confidence']:.1%}"
        )
    
    def _create_error_result(self, error_message: str, symbol: str = "UNKNOWN") -> OrchestrationResult:
        """Create error result"""
        return OrchestrationResult(
            timestamp=datetime.now(),
            symbol=symbol,
            success=False,
            mode=self.config.mode,
            data={},
            errors=[error_message],
            metrics={}
        )
    
    def get_ml_performance_report(self) -> Dict:
        """Get detailed ML performance report"""
        report = {
            'metrics': self.ml_performance_metrics,
            'recent_decisions': [
                {
                    'symbol': d.symbol,
                    'strategy': d.strategy,
                    'confidence': d.confidence,
                    'decision': d.decision,
                    'timestamp': d.timestamp.isoformat()
                }
                for d in self.ml_decisions_history[-10:]  # Last 10 decisions
            ],
            'cache_status': {
                'enabled': self.ml_integrator.enable_caching,
                'ttl_minutes': self.ml_integrator.cache_ttl
            }
        }
        
        return report
    
    def clear_ml_cache(self):
        """Clear ML prediction cache"""
        self.ml_integrator.clear_cache()
        self.logger.info("ML cache cleared")


def create_enhanced_orchestrator(config: Optional[OrchestratorConfig] = None) -> EnhancedTradingOrchestrator:
    """Factory function to create enhanced orchestrator"""
    return EnhancedTradingOrchestrator(config)
