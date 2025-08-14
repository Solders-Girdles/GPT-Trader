"""
Portfolio rebalancing engine with ML integration
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging

from .costs import TransactionCostModel, CostParameters
from .triggers import RebalancingTrigger, ThresholdTrigger, TimeTrigger, RegimeTrigger
from ..ml.portfolio import MLEnhancedAllocator, MarkowitzOptimizer
from ..ml.models import MarketRegimeDetector


@dataclass
class RebalancingConfig:
    """Configuration for rebalancing engine"""
    
    # Rebalancing thresholds
    weight_tolerance: float = 0.05  # 5% weight deviation trigger
    time_interval_days: int = 30  # Rebalance at least monthly
    min_rebalance_value: float = 1000  # Minimum trade size to trigger
    
    # Cost considerations
    include_transaction_costs: bool = True
    max_turnover: float = 0.5  # Maximum 50% turnover per rebalance
    cost_benefit_ratio: float = 2.0  # Benefit must be 2x cost
    
    # Risk constraints
    max_position_change: float = 0.2  # Max 20% change in any position
    emergency_rebalance_threshold: float = 0.3  # 30% deviation triggers immediate rebalance
    
    # ML integration
    use_ml_optimization: bool = True
    regime_aware_rebalancing: bool = True
    adaptive_thresholds: bool = True


class RebalancingEngine:
    """Engine for portfolio rebalancing with ML and cost optimization"""
    
    def __init__(self,
                 config: Optional[RebalancingConfig] = None,
                 allocator: Optional[MLEnhancedAllocator] = None,
                 cost_model: Optional[TransactionCostModel] = None,
                 regime_detector: Optional[MarketRegimeDetector] = None,
                 db_manager=None):
        """Initialize rebalancing engine
        
        Args:
            config: Rebalancing configuration
            allocator: Portfolio allocator
            cost_model: Transaction cost model
            regime_detector: Market regime detector
            db_manager: Database manager
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or RebalancingConfig()
        self.db_manager = db_manager
        
        # Components
        self.allocator = allocator or MLEnhancedAllocator(db_manager=db_manager)
        self.cost_model = cost_model or TransactionCostModel()
        self.regime_detector = regime_detector
        
        # Triggers
        self.triggers = self._initialize_triggers()
        
        # State tracking
        self.current_positions = {}
        self.target_positions = {}
        self.last_rebalance_date = None
        self.rebalance_history = []
        
        # Performance tracking
        self.total_rebalances = 0
        self.total_cost = 0.0
        self.total_turnover = 0.0
        
    def _initialize_triggers(self) -> List[RebalancingTrigger]:
        """Initialize rebalancing triggers
        
        Returns:
            List of triggers
        """
        triggers = []
        
        # Weight threshold trigger
        triggers.append(ThresholdTrigger(
            threshold=self.config.weight_tolerance,
            emergency_threshold=self.config.emergency_rebalance_threshold
        ))
        
        # Time-based trigger
        triggers.append(TimeTrigger(
            interval_days=self.config.time_interval_days
        ))
        
        # Regime change trigger (if available)
        if self.config.regime_aware_rebalancing and self.regime_detector:
            triggers.append(RegimeTrigger(
                regime_detector=self.regime_detector
            ))
        
        return triggers
    
    def check_rebalancing_needed(self,
                                current_positions: Dict[str, float],
                                market_data: Dict[str, pd.DataFrame],
                                current_prices: Dict[str, float]) -> Tuple[bool, str, Dict[str, Any]]:
        """Check if rebalancing is needed
        
        Args:
            current_positions: Current position values
            market_data: Market data for each asset
            current_prices: Current asset prices
            
        Returns:
            Tuple of (needs_rebalancing, reason, details)
        """
        self.current_positions = current_positions
        portfolio_value = sum(current_positions.values())
        
        # Check each trigger
        trigger_results = []
        
        for trigger in self.triggers:
            triggered, urgency, details = trigger.check(
                current_positions=current_positions,
                portfolio_value=portfolio_value,
                last_rebalance_date=self.last_rebalance_date,
                market_data=market_data
            )
            
            if triggered:
                trigger_results.append({
                    'trigger': trigger.__class__.__name__,
                    'urgency': urgency,
                    'details': details
                })
        
        if not trigger_results:
            return False, "No triggers activated", {}
        
        # Find highest urgency trigger
        max_urgency_result = max(trigger_results, key=lambda x: x['urgency'])
        
        # Get target allocation
        universe = list(current_positions.keys())
        target_positions = self.allocator.allocate(
            universe=universe,
            market_data=market_data,
            capital=portfolio_value
        )
        
        # Estimate rebalancing cost
        cost_estimate = self.cost_model.estimate_rebalancing_cost(
            current_positions=current_positions,
            target_positions=target_positions,
            prices=current_prices
        )
        
        # Check cost-benefit ratio
        expected_benefit = self._estimate_rebalancing_benefit(
            current_positions, target_positions, market_data
        )
        
        cost_benefit_ratio = expected_benefit / cost_estimate['total_cost'] if cost_estimate['total_cost'] > 0 else float('inf')
        
        # Decision logic
        needs_rebalancing = False
        reason = ""
        
        if max_urgency_result['urgency'] >= 0.8:  # Emergency rebalancing
            needs_rebalancing = True
            reason = f"Emergency: {max_urgency_result['trigger']}"
        elif cost_benefit_ratio >= self.config.cost_benefit_ratio:
            needs_rebalancing = True
            reason = f"Triggered: {max_urgency_result['trigger']} (CBR: {cost_benefit_ratio:.1f})"
        else:
            reason = f"Triggered but costs too high (CBR: {cost_benefit_ratio:.1f})"
        
        details = {
            'triggers': trigger_results,
            'cost_estimate': cost_estimate,
            'expected_benefit': expected_benefit,
            'cost_benefit_ratio': cost_benefit_ratio,
            'target_positions': target_positions
        }
        
        return needs_rebalancing, reason, details
    
    def execute_rebalancing(self,
                           current_positions: Dict[str, float],
                           target_positions: Dict[str, float],
                           current_prices: Dict[str, float],
                           urgency: str = 'normal') -> Dict[str, Any]:
        """Execute portfolio rebalancing
        
        Args:
            current_positions: Current positions
            target_positions: Target positions
            current_prices: Current prices
            urgency: Execution urgency
            
        Returns:
            Dictionary with rebalancing results
        """
        self.logger.info(f"Executing rebalancing with urgency: {urgency}")
        
        # Calculate required trades
        trades = {}
        for symbol in set(current_positions.keys()) | set(target_positions.keys()):
            current = current_positions.get(symbol, 0.0)
            target = target_positions.get(symbol, 0.0)
            trade = target - current
            
            # Apply minimum trade filter
            if abs(trade) >= self.config.min_rebalance_value:
                # Apply position change limit
                if current > 0:
                    max_change = current * self.config.max_position_change
                    trade = np.clip(trade, -max_change, max_change)
                
                trades[symbol] = trade
        
        # Check turnover constraint
        total_turnover = sum(abs(t) for t in trades.values())
        portfolio_value = sum(current_positions.values())
        turnover_ratio = total_turnover / portfolio_value if portfolio_value > 0 else 0
        
        if turnover_ratio > self.config.max_turnover:
            # Scale down trades to meet turnover constraint
            scale_factor = self.config.max_turnover / turnover_ratio
            trades = {k: v * scale_factor for k, v in trades.items()}
            self.logger.warning(f"Scaled down trades to meet turnover constraint ({turnover_ratio:.1%} -> {self.config.max_turnover:.1%})")
        
        # Optimize execution
        execution_plan = self.cost_model.optimize_trade_execution(trades, urgency)
        
        # Calculate actual costs
        final_positions = current_positions.copy()
        for symbol, trade in trades.items():
            final_positions[symbol] = final_positions.get(symbol, 0.0) + trade
        
        actual_costs = self.cost_model.estimate_rebalancing_cost(
            current_positions=current_positions,
            target_positions=final_positions,
            prices=current_prices
        )
        
        # Update state
        self.current_positions = final_positions
        self.target_positions = target_positions
        self.last_rebalance_date = datetime.now()
        self.total_rebalances += 1
        self.total_cost += actual_costs['total_cost']
        self.total_turnover += actual_costs['total_turnover']
        
        # Create result summary
        result = {
            'timestamp': datetime.now(),
            'trades': trades,
            'execution_plan': execution_plan,
            'costs': actual_costs,
            'turnover_ratio': turnover_ratio,
            'n_trades': len(trades),
            'portfolio_value': portfolio_value,
            'urgency': urgency,
            'success': True
        }
        
        # Record in history
        self._record_rebalancing(result)
        
        return result
    
    def _estimate_rebalancing_benefit(self,
                                     current_positions: Dict[str, float],
                                     target_positions: Dict[str, float],
                                     market_data: Dict[str, pd.DataFrame]) -> float:
        """Estimate benefit of rebalancing
        
        Args:
            current_positions: Current positions
            target_positions: Target positions
            market_data: Market data
            
        Returns:
            Estimated benefit in dollars
        """
        # Simple estimation based on expected improvement in Sharpe ratio
        # In practice, this would use more sophisticated forecasting
        
        portfolio_value = sum(current_positions.values())
        
        # Calculate current and target concentration
        current_weights = {k: v/portfolio_value for k, v in current_positions.items()}
        target_weights = {k: v/portfolio_value for k, v in target_positions.items()}
        
        # Estimate improvement (simplified)
        current_concentration = sum(w**2 for w in current_weights.values())
        target_concentration = sum(w**2 for w in target_weights.values())
        
        # Lower concentration is better (more diversified)
        concentration_improvement = max(0, current_concentration - target_concentration)
        
        # Estimate dollar benefit (rough approximation)
        # Assume 1% improvement in returns for 10% reduction in concentration
        expected_return_improvement = concentration_improvement * 0.1
        annual_benefit = portfolio_value * expected_return_improvement
        
        # Pro-rate to rebalancing frequency
        days_until_next = self.config.time_interval_days
        benefit = annual_benefit * (days_until_next / 365)
        
        return benefit
    
    def get_rebalancing_schedule(self) -> Dict[str, Any]:
        """Get rebalancing schedule and recommendations
        
        Returns:
            Dictionary with schedule information
        """
        schedule = {
            'last_rebalance': self.last_rebalance_date,
            'next_scheduled': None,
            'days_since_last': None,
            'current_regime': None,
            'recommended_frequency': None
        }
        
        if self.last_rebalance_date:
            days_since = (datetime.now() - self.last_rebalance_date).days
            schedule['days_since_last'] = days_since
            schedule['next_scheduled'] = self.last_rebalance_date + timedelta(days=self.config.time_interval_days)
        
        # Get current regime if available
        if self.regime_detector:
            # Would need current market data to detect regime
            pass
        
        # Recommend frequency based on market conditions
        if self.regime_detector and hasattr(self, 'current_regime'):
            regime_frequencies = {
                'bull_quiet': 45,  # Less frequent in stable bull
                'bull_volatile': 21,  # More frequent in volatile bull
                'bear_quiet': 30,  # Monthly in quiet bear
                'bear_volatile': 14,  # Bi-weekly in volatile bear
                'sideways': 30  # Monthly in sideways
            }
            schedule['recommended_frequency'] = regime_frequencies.get(self.current_regime, 30)
        
        return schedule
    
    def analyze_rebalancing_history(self) -> Dict[str, Any]:
        """Analyze historical rebalancing performance
        
        Returns:
            Dictionary with analysis
        """
        if not self.rebalance_history:
            return {'status': 'No rebalancing history'}
        
        # Calculate statistics
        n_rebalances = len(self.rebalance_history)
        total_cost = sum(r['costs']['total_cost'] for r in self.rebalance_history)
        avg_cost = total_cost / n_rebalances if n_rebalances > 0 else 0
        
        avg_turnover = np.mean([r['turnover_ratio'] for r in self.rebalance_history])
        avg_trades = np.mean([r['n_trades'] for r in self.rebalance_history])
        
        # Time between rebalances
        if n_rebalances > 1:
            timestamps = [r['timestamp'] for r in self.rebalance_history]
            intervals = []
            for i in range(1, len(timestamps)):
                interval = (timestamps[i] - timestamps[i-1]).days
                intervals.append(interval)
            avg_interval = np.mean(intervals)
        else:
            avg_interval = self.config.time_interval_days
        
        # Urgency distribution
        urgency_counts = {}
        for record in self.rebalance_history:
            urgency = record.get('urgency', 'normal')
            urgency_counts[urgency] = urgency_counts.get(urgency, 0) + 1
        
        analysis = {
            'total_rebalances': n_rebalances,
            'total_cost': total_cost,
            'avg_cost_per_rebalance': avg_cost,
            'avg_turnover_ratio': avg_turnover,
            'avg_trades_per_rebalance': avg_trades,
            'avg_days_between_rebalances': avg_interval,
            'urgency_distribution': urgency_counts,
            'cost_per_dollar_traded': (total_cost / self.total_turnover) if self.total_turnover > 0 else 0
        }
        
        return analysis
    
    def _record_rebalancing(self, result: Dict[str, Any]):
        """Record rebalancing event
        
        Args:
            result: Rebalancing result
        """
        self.rebalance_history.append(result)
        
        # Keep only last 100 records
        if len(self.rebalance_history) > 100:
            self.rebalance_history = self.rebalance_history[-100:]
        
        # Store in database if available
        if self.db_manager:
            try:
                import json
                self.db_manager.execute(
                    """INSERT INTO rebalancing_events 
                       (timestamp, trades, costs, turnover, urgency, details)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (
                        result['timestamp'],
                        json.dumps(result['trades']),
                        result['costs']['total_cost'],
                        result['turnover_ratio'],
                        result['urgency'],
                        json.dumps(result)
                    )
                )
            except Exception as e:
                self.logger.error(f"Error storing rebalancing event: {e}")