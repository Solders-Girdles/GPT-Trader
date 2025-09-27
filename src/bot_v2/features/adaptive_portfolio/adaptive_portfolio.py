"""
Main adaptive portfolio management module.

Provides the primary interface for adaptive portfolio management with 
configuration-driven behavior based on portfolio size.
"""

try:
    import pandas as pd
    import numpy as np
    HAS_PANDAS = True
    DataFrame = pd.DataFrame
    Timestamp = pd.Timestamp
except ImportError:
    HAS_PANDAS = False
    from typing import Any
    DataFrame = Any
    Timestamp = Any

from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple

from .types import (
    PortfolioTier, AdaptiveResult, TierConfig, PortfolioSnapshot,
    PositionInfo, TradingSignal, BacktestMetrics, ValidationResult
)
from .config_manager import load_portfolio_config, get_current_tier, validate_portfolio_config
from .tier_manager import TierManager
from .risk_manager import AdaptiveRiskManager
from .strategy_selector import StrategySelector
from ...data_providers import DataProvider, get_data_provider


class AdaptivePortfolioManager:
    """
    Main class for adaptive portfolio management.
    
    Automatically adjusts strategy, risk, and position sizing based on 
    portfolio size and configuration.
    """
    
    def __init__(
        self, 
        config_path: Optional[str] = None,
        data_provider: Optional[DataProvider] = None,
        prefer_real_data: bool = True
    ):
        """
        Initialize with optional custom config path and data provider.
        
        Args:
            config_path: Optional path to custom configuration file
            data_provider: Optional data provider instance. If None, will create one automatically
            prefer_real_data: If True and data_provider is None, prefer real data over mock
        """
        self.config_path = config_path
        self.config = load_portfolio_config(config_path)
        
        # Initialize data provider
        if data_provider is None:
            if prefer_real_data:
                self.data_provider = get_data_provider('yfinance')
                self.data_provider_type = 'yfinance'
            else:
                self.data_provider = get_data_provider('mock')
                self.data_provider_type = 'mock'
        else:
            self.data_provider = data_provider
            self.data_provider_type = 'custom'
        
        self.tier_manager = TierManager(self.config)
        self.risk_manager = AdaptiveRiskManager(self.config)
        self.strategy_selector = StrategySelector(self.config, self.data_provider)
    
    def analyze_portfolio(
        self, 
        current_capital: float,
        positions: Optional[List[PositionInfo]] = None,
        market_data: Optional[Dict[str, DataFrame]] = None
    ) -> AdaptiveResult:
        """
        Analyze current portfolio and generate adaptive recommendations.
        
        Args:
            current_capital: Current total portfolio value
            positions: Current positions (if any)
            market_data: Optional market data for analysis
            
        Returns:
            AdaptiveResult with tier-appropriate recommendations
        """
        # Determine current tier
        current_tier_name = get_current_tier(current_capital, self.config_path)
        current_tier = PortfolioTier(current_tier_name)
        tier_config = self.config.tiers[current_tier_name]
        
        # Create portfolio snapshot
        if positions is None:
            positions = []
        
        portfolio_snapshot = self._create_portfolio_snapshot(
            current_capital, positions, current_tier
        )
        
        # Check for tier transition
        tier_transition_needed, tier_transition_target = self._check_tier_transition(
            current_capital, current_tier
        )
        
        # Generate trading signals based on tier
        signals = self._generate_tier_appropriate_signals(
            tier_config, portfolio_snapshot, market_data
        )
        
        # Calculate risk metrics
        risk_metrics = self.risk_manager.calculate_risk_metrics(
            portfolio_snapshot, tier_config
        )
        
        # Generate recommendations and warnings
        recommendations = self._generate_recommendations(
            portfolio_snapshot, tier_config, signals, risk_metrics
        )
        
        warnings = self._generate_warnings(
            portfolio_snapshot, tier_config, risk_metrics
        )
        
        return AdaptiveResult(
            current_tier=current_tier,
            tier_config=tier_config,
            portfolio_snapshot=portfolio_snapshot,
            signals=signals,
            risk_metrics=risk_metrics,
            tier_transition_needed=tier_transition_needed,
            tier_transition_target=tier_transition_target,
            recommended_actions=recommendations,
            warnings=warnings,
            timestamp=datetime.now()
        )
    
    def run_adaptive_backtest(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        initial_capital: float = 1000
    ) -> BacktestMetrics:
        """
        Run backtest with adaptive tier management.
        
        Portfolio grows/shrinks and adapts behavior accordingly.
        """
        if not HAS_PANDAS:
            raise ImportError(
                "Backtesting requires pandas. "
                "Install with: pip install pandas"
            )
        
        # Download market data using data provider
        data = {}
        for symbol in symbols:
            try:
                df = self.data_provider.get_historical_data(
                    symbol, start=start_date, end=end_date
                )
                data[symbol] = df
            except Exception as e:
                raise ValueError(f"Failed to get data for {symbol}: {e}")
        
        # Initialize backtest state
        current_capital = initial_capital
        positions = []
        trades = []
        daily_values = []
        tier_transitions = []
        
        # Get date range
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        dates = pd.date_range(start_dt, end_dt, freq='D')
        
        for current_date in dates:
            # Skip weekends
            if current_date.weekday() >= 5:
                continue
            
            # Update position values
            current_capital, positions = self._update_positions(
                positions, data, current_date
            )
            
            # Analyze portfolio for current date
            result = self.analyze_portfolio(current_capital, positions)
            
            # Check for tier transition
            if result.tier_transition_needed:
                tier_transitions.append({
                    'date': current_date,
                    'from_tier': result.current_tier.value,
                    'to_tier': result.tier_transition_target.value if result.tier_transition_target else None,
                    'capital': current_capital
                })
            
            # Execute signals
            new_trades = self._execute_signals(
                result.signals, data, current_date, current_capital
            )
            trades.extend(new_trades)
            
            # Update positions with new trades
            positions = self._update_positions_with_trades(positions, new_trades)
            
            # Record daily value
            daily_values.append({
                'date': current_date,
                'total_value': current_capital,
                'tier': result.current_tier.value
            })
        
        # Calculate metrics
        return self._calculate_backtest_metrics(
            daily_values, trades, tier_transitions, initial_capital
        )
    
    def _create_portfolio_snapshot(
        self, 
        total_value: float, 
        positions: List[PositionInfo], 
        current_tier: PortfolioTier
    ) -> PortfolioSnapshot:
        """Create snapshot of current portfolio state."""
        
        cash = total_value - sum(pos.position_value for pos in positions)
        daily_pnl = sum(pos.unrealized_pnl for pos in positions)
        daily_pnl_pct = daily_pnl / total_value * 100 if total_value > 0 else 0
        
        # Calculate sector exposures (simplified)
        sector_exposures = {}  # Would need sector mapping in real implementation
        
        largest_position_pct = 0
        if positions:
            largest_position_pct = max(pos.position_value / total_value * 100 for pos in positions)
        
        return PortfolioSnapshot(
            total_value=total_value,
            cash=cash,
            positions=positions,
            daily_pnl=daily_pnl,
            daily_pnl_pct=daily_pnl_pct,
            quarterly_pnl_pct=0,  # Would calculate from historical data
            current_tier=current_tier,
            positions_count=len(positions),
            largest_position_pct=largest_position_pct,
            sector_exposures=sector_exposures
        )
    
    def _check_tier_transition(
        self, 
        current_capital: float, 
        current_tier: PortfolioTier
    ) -> Tuple[bool, Optional[PortfolioTier]]:
        """Check if portfolio needs to transition to different tier."""
        
        new_tier_name = get_current_tier(current_capital, self.config_path)
        new_tier = PortfolioTier(new_tier_name)
        
        if new_tier != current_tier:
            return True, new_tier
        
        return False, None
    
    def _generate_tier_appropriate_signals(
        self,
        tier_config: TierConfig,
        portfolio_snapshot: PortfolioSnapshot,
        market_data: Optional[Dict[str, DataFrame]]
    ) -> List[TradingSignal]:
        """Generate trading signals appropriate for current tier."""
        
        return self.strategy_selector.generate_signals(
            tier_config, portfolio_snapshot, market_data
        )
    
    def _generate_recommendations(
        self,
        portfolio_snapshot: PortfolioSnapshot,
        tier_config: TierConfig,
        signals: List[TradingSignal],
        risk_metrics: Dict[str, float]
    ) -> List[str]:
        """Generate human-readable recommendations."""
        
        recommendations = []
        
        # Position count recommendations
        current_positions = portfolio_snapshot.positions_count
        target_positions = tier_config.positions.target_positions
        
        if current_positions < target_positions:
            recommendations.append(
                f"Consider adding {target_positions - current_positions} more positions "
                f"to reach target of {target_positions}"
            )
        elif current_positions > tier_config.positions.max_positions:
            recommendations.append(
                f"Consider reducing positions to {tier_config.positions.max_positions} maximum"
            )
        
        # Cash allocation
        cash_pct = portfolio_snapshot.cash / portfolio_snapshot.total_value * 100
        if cash_pct > 20:
            recommendations.append(f"High cash allocation ({cash_pct:.1f}%) - consider deploying capital")
        
        # Risk recommendations
        if risk_metrics.get('daily_risk_pct', 0) > tier_config.risk.daily_limit_pct:
            recommendations.append("Daily risk exceeds tier limits - consider reducing position sizes")
        
        return recommendations
    
    def _generate_warnings(
        self,
        portfolio_snapshot: PortfolioSnapshot,
        tier_config: TierConfig,
        risk_metrics: Dict[str, float]
    ) -> List[str]:
        """Generate warning messages for risk situations."""
        
        warnings = []
        
        # Risk warnings
        if portfolio_snapshot.largest_position_pct > 25:
            warnings.append(
                f"Largest position is {portfolio_snapshot.largest_position_pct:.1f}% of portfolio - "
                "consider diversifying"
            )
        
        if risk_metrics.get('daily_risk_pct', 0) > tier_config.risk.daily_limit_pct * 0.8:
            warnings.append("Approaching daily risk limit")
        
        # PDT warnings for small accounts
        if tier_config.trading.pdt_compliant and portfolio_snapshot.total_value < 25000:
            warnings.append("Account under $25K - limit day trades to avoid PDT violations")
        
        return warnings
    
    def _update_positions(
        self, 
        positions: List[PositionInfo], 
        data: Dict[str, DataFrame], 
        current_date: Timestamp
    ) -> Tuple[float, List[PositionInfo]]:
        """Update position values with current market prices."""
        
        updated_positions = []
        total_value = 0
        
        for pos in positions:
            if pos.symbol in data and current_date in data[pos.symbol].index:
                current_price = data[pos.symbol].loc[current_date, 'Close']
                position_value = pos.shares * current_price
                unrealized_pnl = position_value - (pos.shares * pos.entry_price)
                unrealized_pnl_pct = unrealized_pnl / (pos.shares * pos.entry_price) * 100
                
                updated_pos = PositionInfo(
                    symbol=pos.symbol,
                    shares=pos.shares,
                    entry_price=pos.entry_price,
                    current_price=current_price,
                    position_value=position_value,
                    unrealized_pnl=unrealized_pnl,
                    unrealized_pnl_pct=unrealized_pnl_pct,
                    days_held=pos.days_held + 1,
                    stop_loss_price=pos.stop_loss_price
                )
                updated_positions.append(updated_pos)
                total_value += position_value
        
        return total_value, updated_positions
    
    def _execute_signals(
        self,
        signals: List[TradingSignal],
        data: Dict[str, DataFrame],
        current_date: Timestamp,
        available_capital: float
    ) -> List[Dict]:
        """Execute trading signals and return trade records."""
        
        trades = []
        
        for signal in signals:
            if signal.action == "BUY" and signal.symbol in data:
                if current_date in data[signal.symbol].index:
                    price = data[signal.symbol].loc[current_date, 'Close']
                    shares = int(signal.target_position_size / price)
                    
                    if shares > 0 and shares * price <= available_capital:
                        trades.append({
                            'date': current_date,
                            'symbol': signal.symbol,
                            'action': 'BUY',
                            'shares': shares,
                            'price': price,
                            'value': shares * price,
                            'strategy': signal.strategy_source
                        })
        
        return trades
    
    def _update_positions_with_trades(
        self, 
        positions: List[PositionInfo], 
        new_trades: List[Dict]
    ) -> List[PositionInfo]:
        """Update positions list with new trades."""
        
        # This is a simplified implementation
        # Real implementation would handle position updates, sells, etc.
        updated_positions = positions.copy()
        
        for trade in new_trades:
            if trade['action'] == 'BUY':
                new_position = PositionInfo(
                    symbol=trade['symbol'],
                    shares=trade['shares'],
                    entry_price=trade['price'],
                    current_price=trade['price'],
                    position_value=trade['value'],
                    unrealized_pnl=0,
                    unrealized_pnl_pct=0,
                    days_held=0
                )
                updated_positions.append(new_position)
        
        return updated_positions
    
    def _calculate_backtest_metrics(
        self,
        daily_values: List[Dict],
        trades: List[Dict],
        tier_transitions: List[Dict],
        initial_capital: float
    ) -> BacktestMetrics:
        """Calculate comprehensive backtest metrics."""
        
        if not daily_values:
            return BacktestMetrics(
                total_return_pct=0, annualized_return_pct=0, max_drawdown_pct=0,
                sharpe_ratio=0, win_rate_pct=0, avg_trade_return_pct=0,
                total_trades=0, tier_transitions=0, final_tier=PortfolioTier.MICRO,
                tier_performance={}
            )
        
        # Calculate returns
        final_value = daily_values[-1]['total_value']
        total_return_pct = (final_value - initial_capital) / initial_capital * 100
        
        # Calculate annualized return
        days = len(daily_values)
        years = days / 365.25
        annualized_return_pct = ((final_value / initial_capital) ** (1/years) - 1) * 100 if years > 0 else 0
        
        # Calculate max drawdown
        peak = initial_capital
        max_drawdown_pct = 0
        for day in daily_values:
            value = day['total_value']
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak * 100
            if drawdown > max_drawdown_pct:
                max_drawdown_pct = drawdown
        
        # Calculate basic metrics
        sharpe_ratio = 0  # Would need daily returns for proper calculation
        win_rate_pct = 0  # Would need individual trade P&L
        avg_trade_return_pct = 0
        
        final_tier = PortfolioTier(daily_values[-1]['tier']) if daily_values else PortfolioTier.MICRO
        
        return BacktestMetrics(
            total_return_pct=total_return_pct,
            annualized_return_pct=annualized_return_pct,
            max_drawdown_pct=max_drawdown_pct,
            sharpe_ratio=sharpe_ratio,
            win_rate_pct=win_rate_pct,
            avg_trade_return_pct=avg_trade_return_pct,
            total_trades=len(trades),
            tier_transitions=len(tier_transitions),
            final_tier=final_tier,
            tier_performance={}
        )


# Convenience functions for external use

def run_adaptive_strategy(
    current_capital: float,
    symbols: Optional[List[str]] = None,
    positions: Optional[List[PositionInfo]] = None,
    config_path: Optional[str] = None,
    data_provider: Optional[DataProvider] = None,
    prefer_real_data: bool = True
) -> AdaptiveResult:
    """
    Main entry point for adaptive portfolio analysis.
    
    Args:
        current_capital: Current portfolio value
        symbols: Optional list of symbols to analyze
        positions: Current positions
        config_path: Optional custom config path
        data_provider: Optional data provider instance
        prefer_real_data: If True, prefer real data over mock when creating provider
        
    Returns:
        AdaptiveResult with tier-appropriate recommendations
    """
    manager = AdaptivePortfolioManager(config_path, data_provider, prefer_real_data)
    return manager.analyze_portfolio(current_capital, positions)


def run_adaptive_backtest(
    symbols: List[str],
    start_date: str,
    end_date: str,
    initial_capital: float = 1000,
    config_path: Optional[str] = None,
    data_provider: Optional[DataProvider] = None,
    prefer_real_data: bool = True
) -> BacktestMetrics:
    """
    Run adaptive backtest with tier transitions.
    
    Args:
        symbols: List of symbols to trade
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        initial_capital: Starting capital
        config_path: Optional custom config path
        data_provider: Optional data provider instance
        prefer_real_data: If True, prefer real data over mock when creating provider
        
    Returns:
        BacktestMetrics with performance across tiers
    """
    manager = AdaptivePortfolioManager(config_path, data_provider, prefer_real_data)
    return manager.run_adaptive_backtest(symbols, start_date, end_date, initial_capital)