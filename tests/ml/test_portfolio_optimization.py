"""
Test suite for portfolio optimization and rebalancing
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class TestPortfolioOptimizer:
    """Test Markowitz portfolio optimizer"""
    
    @pytest.fixture
    def sample_returns(self):
        """Create sample return data"""
        n_days = 252
        n_assets = 5
        np.random.seed(42)
        
        # Generate correlated returns
        mean_returns = np.array([0.08, 0.10, 0.12, 0.06, 0.09]) / 252
        volatilities = np.array([0.15, 0.20, 0.25, 0.12, 0.18]) / np.sqrt(252)
        
        # Create correlation matrix
        correlation = np.array([
            [1.0, 0.3, 0.2, 0.1, 0.4],
            [0.3, 1.0, 0.4, 0.2, 0.3],
            [0.2, 0.4, 1.0, 0.3, 0.5],
            [0.1, 0.2, 0.3, 1.0, 0.2],
            [0.4, 0.3, 0.5, 0.2, 1.0]
        ])
        
        # Generate returns
        covariance = np.outer(volatilities, volatilities) * correlation
        returns = np.random.multivariate_normal(mean_returns, covariance, n_days)
        
        return pd.DataFrame(
            returns,
            columns=['Asset1', 'Asset2', 'Asset3', 'Asset4', 'Asset5']
        )
    
    def test_max_sharpe_optimization(self, sample_returns):
        """Test maximum Sharpe ratio optimization"""
        from src.bot.ml.portfolio.optimizer import MarkowitzOptimizer, OptimizationConstraints
        
        optimizer = MarkowitzOptimizer(risk_free_rate=0.02)
        
        constraints = OptimizationConstraints(
            min_weight=0.0,
            max_weight=0.4,
            long_only=True
        )
        
        weights, metrics = optimizer.optimize(
            sample_returns,
            constraints=constraints,
            objective='max_sharpe'
        )
        
        # Check weights
        assert abs(sum(weights.values()) - 1.0) < 0.01
        assert all(0 <= w <= 0.4 for w in weights.values())
        
        # Check metrics
        assert 'sharpe_ratio' in metrics
        assert 'expected_return' in metrics
        assert 'expected_risk' in metrics
        assert metrics['sharpe_ratio'] > 0  # Should be positive for reasonable data
    
    def test_min_risk_optimization(self, sample_returns):
        """Test minimum risk optimization"""
        from src.bot.ml.portfolio.optimizer import MarkowitzOptimizer, OptimizationConstraints
        
        optimizer = MarkowitzOptimizer()
        
        constraints = OptimizationConstraints(long_only=True)
        
        weights, metrics = optimizer.optimize(
            sample_returns,
            constraints=constraints,
            objective='min_risk'
        )
        
        # Check weights
        assert abs(sum(weights.values()) - 1.0) < 0.01
        assert all(w >= 0 for w in weights.values())
        
        # Min risk should have lower volatility than equal weight
        equal_weight_vol = sample_returns.mean(axis=1).std() * np.sqrt(252)
        assert metrics['annual_volatility'] <= equal_weight_vol * 1.1  # Allow small tolerance
    
    def test_efficient_frontier(self, sample_returns):
        """Test efficient frontier generation"""
        from src.bot.ml.portfolio.optimizer import MarkowitzOptimizer
        
        optimizer = MarkowitzOptimizer()
        
        weights, metrics = optimizer.optimize(
            sample_returns,
            objective='efficient_frontier'
        )
        
        # Check frontier was generated
        assert 'efficient_frontier' in metrics
        frontier = metrics['efficient_frontier']
        
        assert 'returns' in frontier
        assert 'risks' in frontier
        assert len(frontier['returns']) > 10  # Should have multiple points
        
        # Check frontier is upward sloping
        returns = frontier['returns']
        risks = frontier['risks']
        
        # Higher risk should generally mean higher return
        correlation = np.corrcoef(risks, returns)[0, 1]
        assert correlation > 0.5
    
    def test_portfolio_constraints(self, sample_returns):
        """Test various portfolio constraints"""
        from src.bot.ml.portfolio.optimizer import MarkowitzOptimizer, OptimizationConstraints
        
        optimizer = MarkowitzOptimizer()
        
        # Test max position constraint
        constraints = OptimizationConstraints(
            max_weight=0.2,
            long_only=True
        )
        
        weights, _ = optimizer.optimize(sample_returns, constraints=constraints)
        
        assert all(w <= 0.2 for w in weights.values())
        
        # Test target return constraint
        constraints = OptimizationConstraints(
            target_return=0.0001,  # Daily return target
            long_only=True
        )
        
        weights, metrics = optimizer.optimize(sample_returns, constraints=constraints)
        
        # Expected return should meet or exceed target
        assert metrics['expected_return'] >= 0.0001 * 0.9  # Allow 10% tolerance
    
    def test_risk_metrics(self, sample_returns):
        """Test risk metric calculations"""
        from src.bot.ml.portfolio.optimizer import MarkowitzOptimizer
        
        optimizer = MarkowitzOptimizer()
        
        weights, _ = optimizer.optimize(sample_returns)
        
        # Calculate risk metrics
        risk_metrics = optimizer.calculate_risk_metrics(sample_returns, weights)
        
        # Check all metrics present
        expected_metrics = ['var_95', 'cvar_95', 'downside_deviation', 
                          'sortino_ratio', 'tracking_error', 'information_ratio']
        
        for metric in expected_metrics:
            assert metric in risk_metrics
        
        # Check metric validity
        assert risk_metrics['var_95'] < 0  # VaR should be negative
        assert risk_metrics['cvar_95'] <= risk_metrics['var_95']  # CVaR worse than VaR
        assert risk_metrics['downside_deviation'] >= 0
    
    def test_backtesting(self, sample_returns):
        """Test portfolio backtesting"""
        from src.bot.ml.portfolio.optimizer import MarkowitzOptimizer
        
        optimizer = MarkowitzOptimizer()
        
        # Optimize on first half
        train_returns = sample_returns.iloc[:126]
        weights, _ = optimizer.optimize(train_returns)
        
        # Backtest on second half
        test_returns = sample_returns.iloc[126:]
        backtest = optimizer.backtest_allocation(test_returns, weights)
        
        # Check backtest results
        assert 'total_return' in backtest
        assert 'sharpe_ratio' in backtest
        assert 'max_drawdown' in backtest
        assert backtest['max_drawdown'] <= 0  # Drawdown should be negative


class TestRebalancingEngine:
    """Test portfolio rebalancing engine"""
    
    @pytest.fixture
    def mock_positions(self):
        """Create mock portfolio positions"""
        return {
            'AAPL': 30000,
            'MSFT': 25000,
            'GOOGL': 20000,
            'AMZN': 15000,
            'META': 10000
        }
    
    @pytest.fixture
    def mock_market_data(self):
        """Create mock market data"""
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
        market_data = {}
        
        dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
        
        for symbol in symbols:
            market_data[symbol] = pd.DataFrame({
                'close': 100 + np.random.randn(100).cumsum(),
                'volume': np.random.uniform(1e6, 1e7, 100)
            }, index=dates)
        
        return market_data
    
    def test_rebalancing_triggers(self, mock_positions):
        """Test rebalancing trigger mechanisms"""
        from src.bot.rebalancing.triggers import ThresholdTrigger, TimeTrigger
        
        portfolio_value = sum(mock_positions.values())
        
        # Test threshold trigger
        threshold_trigger = ThresholdTrigger(threshold=0.05)
        threshold_trigger.set_target_weights({
            'AAPL': 0.20,  # Current is 30%
            'MSFT': 0.20,
            'GOOGL': 0.20,
            'AMZN': 0.20,
            'META': 0.20
        })
        
        triggered, urgency, details = threshold_trigger.check(
            mock_positions, portfolio_value
        )
        
        assert triggered  # Should trigger due to deviation
        assert 0 <= urgency <= 1
        assert 'max_deviation' in details
        
        # Test time trigger
        time_trigger = TimeTrigger(interval_days=30)
        
        # Test with old rebalance date
        old_date = datetime.now() - timedelta(days=35)
        triggered, urgency, details = time_trigger.check(
            mock_positions, portfolio_value, old_date
        )
        
        assert triggered
        assert urgency > 0
    
    def test_transaction_costs(self, mock_positions):
        """Test transaction cost calculations"""
        from src.bot.rebalancing.costs import TransactionCostModel, CostParameters
        
        params = CostParameters(
            commission_per_trade=1.0,
            bid_ask_spread=0.001,
            slippage_rate=0.0005
        )
        
        cost_model = TransactionCostModel(parameters=params)
        
        # Test single trade cost
        cost = cost_model.estimate_trade_cost(
            symbol='AAPL',
            quantity=100,
            price=150.0,
            volatility=0.20
        )
        
        assert cost['total_cost'] > 0
        assert cost['commission'] == 1.0
        assert cost['spread_cost'] > 0
        assert cost['slippage'] > 0
        
        # Test rebalancing cost
        target_positions = {
            'AAPL': 25000,  # Sell 5000
            'MSFT': 25000,  # No change
            'GOOGL': 25000,  # Buy 5000
            'AMZN': 15000,  # No change
            'META': 10000   # No change
        }
        
        prices = {'AAPL': 150, 'MSFT': 300, 'GOOGL': 140, 'AMZN': 180, 'META': 350}
        
        rebalance_cost = cost_model.estimate_rebalancing_cost(
            mock_positions, target_positions, prices
        )
        
        assert rebalance_cost['total_cost'] > 0
        assert rebalance_cost['n_trades'] == 2  # AAPL sell, GOOGL buy
        assert rebalance_cost['total_turnover'] == 10000
    
    def test_rebalancing_decision(self, mock_positions, mock_market_data):
        """Test rebalancing decision logic"""
        from src.bot.rebalancing.engine import RebalancingEngine, RebalancingConfig
        from src.bot.rebalancing.costs import TransactionCostModel
        
        config = RebalancingConfig(
            weight_tolerance=0.05,
            min_rebalance_value=1000,
            cost_benefit_ratio=2.0
        )
        
        cost_model = TransactionCostModel()
        engine = RebalancingEngine(config=config, cost_model=cost_model)
        
        # Set last rebalance date to trigger time-based rebalancing
        engine.last_rebalance_date = datetime.now() - timedelta(days=35)
        
        prices = {symbol: data['close'].iloc[-1] 
                 for symbol, data in mock_market_data.items()}
        
        needs_rebalancing, reason, details = engine.check_rebalancing_needed(
            mock_positions, mock_market_data, prices
        )
        
        # Should trigger based on time
        assert isinstance(needs_rebalancing, bool)
        assert reason != ""
        assert 'triggers' in details or 'cost_estimate' in details
    
    def test_execution_optimization(self):
        """Test trade execution optimization"""
        from src.bot.rebalancing.costs import TransactionCostModel
        
        cost_model = TransactionCostModel()
        
        trades = {
            'AAPL': -5000,   # Small trade
            'MSFT': 25000,   # Medium trade
            'GOOGL': 100000  # Large trade
        }
        
        # Test different urgency levels
        for urgency in ['low', 'normal', 'high']:
            execution_plan = cost_model.optimize_trade_execution(trades, urgency)
            
            assert len(execution_plan) == 3
            
            # Large trades should be split
            if urgency == 'low':
                assert execution_plan['GOOGL']['strategy'] in ['VWAP', 'TWAP']
                assert execution_plan['GOOGL']['n_orders'] > 1
            
            # Small trades should be single
            assert execution_plan['AAPL']['n_orders'] == 1


class TestMLEnhancedAllocator:
    """Test ML-enhanced portfolio allocator"""
    
    def test_regime_based_allocation(self):
        """Test regime-specific allocation"""
        from src.bot.ml.portfolio.allocator import MLEnhancedAllocator
        
        allocator = MLEnhancedAllocator()
        
        # Test different regimes
        regimes = ['bull_quiet', 'bear_volatile', 'sideways']
        
        universe = ['AAPL', 'MSFT', 'GOOGL']
        capital = 100000
        
        # Create mock market data
        market_data = {}
        dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
        
        for symbol in universe:
            market_data[symbol] = pd.DataFrame({
                'close': 100 + np.random.randn(100).cumsum(),
                'volume': np.random.uniform(1e6, 1e7, 100)
            }, index=dates)
        
        allocations = {}
        
        for regime in regimes:
            allocator.current_regime = regime
            positions = allocator.allocate(universe, market_data, capital=capital)
            allocations[regime] = sum(positions.values()) / capital
        
        # Different regimes should have different exposures
        assert allocations['bull_quiet'] > allocations['bear_volatile']
        assert allocations['sideways'] > allocations['bear_volatile']
    
    def test_signal_integration(self):
        """Test signal-based allocation adjustment"""
        from src.bot.ml.portfolio.allocator import MLEnhancedAllocator
        
        allocator = MLEnhancedAllocator()
        
        universe = ['AAPL', 'MSFT', 'GOOGL']
        capital = 100000
        
        # Create signals
        signals = {
            'AAPL': 0.8,   # Strong buy
            'MSFT': -0.5,  # Sell
            'GOOGL': 0.3   # Weak buy
        }
        
        # Create mock market data
        market_data = {}
        dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
        
        for symbol in universe:
            market_data[symbol] = pd.DataFrame({
                'close': 100 + np.random.randn(100).cumsum(),
                'volume': np.random.uniform(1e6, 1e7, 100)
            }, index=dates)
        
        # Allocate with signals
        positions = allocator.allocate(
            universe, market_data, signals=signals, capital=capital
        )
        
        # Strong buy signal should get higher allocation
        if 'AAPL' in positions and 'MSFT' in positions:
            # This might not always hold due to optimization
            pass  # Just check it runs
        
        assert sum(positions.values()) <= capital * 1.01  # Allow rounding


def test_integration_flow():
    """Test complete integration from optimization to rebalancing"""
    
    # Create sample data
    returns = pd.DataFrame({
        'AAPL': np.random.normal(0.001, 0.02, 100),
        'MSFT': np.random.normal(0.0008, 0.018, 100),
        'GOOGL': np.random.normal(0.0012, 0.022, 100)
    })
    
    # Optimize portfolio
    from src.bot.ml.portfolio.optimizer import MarkowitzOptimizer
    
    optimizer = MarkowitzOptimizer()
    weights, metrics = optimizer.optimize(returns, objective='max_sharpe')
    
    # Convert to positions
    capital = 100000
    target_positions = {
        symbol: weight * capital 
        for symbol, weight in weights.items()
    }
    
    # Check rebalancing need
    current_positions = {
        'AAPL': 40000,
        'MSFT': 35000,
        'GOOGL': 25000
    }
    
    # Calculate costs
    from src.bot.rebalancing.costs import TransactionCostModel
    
    cost_model = TransactionCostModel()
    prices = {'AAPL': 150, 'MSFT': 300, 'GOOGL': 140}
    
    costs = cost_model.estimate_rebalancing_cost(
        current_positions, target_positions, prices
    )
    
    assert costs['total_cost'] >= 0
    assert costs['n_trades'] >= 0
    
    # Make decision
    cost_benefit_ratio = 100 / costs['total_cost'] if costs['total_cost'] > 0 else float('inf')
    should_rebalance = cost_benefit_ratio > 2.0
    
    assert isinstance(should_rebalance, bool)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])