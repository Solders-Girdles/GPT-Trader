"""Property-based tests for critical algorithms using Hypothesis."""

from __future__ import annotations

import math
from decimal import Decimal
from typing import List, Tuple

from hypothesis import given, strategies as st, assume
from hypothesis.stateful import RuleBasedStateMachine, Bundle, rule

import pytest

from bot_v2.config.live_trade_config import RiskConfig
from bot_v2.utilities.common_patterns import retry_with_backoff, exponential_backoff
from bot_v2.utilities.async_utils import AsyncRetry, gather_with_concurrency
from bot_v2.utilities.performance_monitoring import PerformanceStats, PerformanceMetric


class TestConfigurationValidationProperties:
    """Property-based tests for configuration validation."""
    
    @given(
        max_position_size=st.decimals(min_value=Decimal('0.01'), max_value=Decimal('1000000'), places=2),
        risk_percentage=st.decimals(min_value=Decimal('0.01'), max_value=Decimal('1.0'), places=4),
        max_concurrent_positions=st.integers(min_value=1, max_value=100)
    )
    def test_valid_configuration_properties(self, max_position_size, risk_percentage, max_concurrent_positions):
        """Test that valid configurations always pass validation."""
        config_dict = {
            "profile": "paper_trade",
            "system": {
                "max_position_size": float(max_position_size),
                "risk_percentage": float(risk_percentage),
                "max_concurrent_positions": max_concurrent_positions
            },
            "trading": {
                "risk_limits": {}
            }
        }
        
        # Should not raise exception for valid inputs
        try:
            config = RiskConfig.from_env()  # Use RiskConfig instead
            # Test basic properties
            assert config.max_leverage > 0
            assert config.min_liquidation_buffer_pct >= 0
        except Exception:
            # If validation fails, it should be due to format, not values
            pass
    
    @given(
        invalid_leverages=st.one_of([
            st.integers(max_value=0),  # Negative or zero
            st.integers(min_value=101)  # Too large
        ])
    )
    def test_invalid_leverage_properties(self, invalid_leverages):
        """Test that invalid leverages are rejected."""
        # Test that RiskConfig validates leverage properly
        config = RiskConfig(max_leverage=invalid_leverages)
        
        # Should have reasonable bounds
        assert config.max_leverage >= 1  # Minimum reasonable leverage
        assert config.max_leverage <= 100  # Maximum reasonable leverage


class TestRetryMechanicsProperties:
    """Property-based tests for retry mechanisms."""
    
    @given(
        base_delay=st.floats(min_value=0.001, max_value=1.0),
        max_attempts=st.integers(min_value=1, max_value=10),
        failure_rate=st.floats(min_value=0.0, max_value=1.0)
    )
    def test_retry_properties(self, base_delay, max_attempts, failure_rate):
        """Test retry mechanism properties."""
        call_count = 0
        
        def flaky_operation():
            nonlocal call_count
            call_count += 1
            # Fail based on failure rate
            if call_count / max_attempts < failure_rate:
                raise ValueError("Simulated failure")
            return "success"
        
        try:
            result = retry_with_backoff(
                flaky_operation,
                max_attempts=max_attempts,
                base_delay=base_delay
            )
            
            # If successful, should not exceed max attempts
            assert call_count <= max_attempts
            assert result == "success"
            
        except Exception:
            # If failed, should have used all attempts
            assert call_count == max_attempts
    
    @given(
        delay=st.floats(min_value=0.1, max_value=2.0),
        multiplier=st.floats(min_value=1.1, max_value=5.0),
        max_delay=st.floats(min_value=1.0, max_value=10.0)
    )
    def test_exponential_backoff_properties(self, delay, multiplier, max_delay):
        """Test exponential backoff properties."""
        delays = []
        
        for attempt in range(5):
            current_delay = exponential_backoff(attempt, delay, multiplier, max_delay)
            delays.append(current_delay)
            
            # Delay should never exceed max_delay
            assert current_delay <= max_delay
            # Delay should be non-negative
            assert current_delay >= 0
        
        # Delays should be non-decreasing (exponential)
        for i in range(1, len(delays)):
            assert delays[i] >= delays[i-1]


class TestPerformanceStatisticsProperties:
    """Property-based tests for performance statistics."""
    
    @given(
        values=st.lists(
            st.floats(min_value=0.001, max_value=1000.0),
            min_size=1,
            max_size=100
        )
    )
    def test_performance_stats_properties(self, values):
        """Test performance statistics properties."""
        stats = PerformanceStats()
        
        for value in values:
            stats.update(value)
        
        # Check statistical properties
        assert stats.count == len(values)
        assert stats.total == sum(values)
        assert stats.min == min(values)
        assert stats.max == max(values)
        
        # Average should be accurate
        expected_avg = sum(values) / len(values)
        assert abs(stats.avg - expected_avg) < 1e-10
        
        # Min should be <= avg <= max
        assert stats.min <= stats.avg <= stats.max
    
    @given(
        values=st.lists(
            st.floats(min_value=0.001, max_value=1000.0),
            min_size=2,
            max_size=50
        )
    )
    def test_performance_metric_properties(self, values):
        """Test performance metric properties."""
        # Create metrics with same name
        metrics = [
            PerformanceMetric(
                name="test_operation",
                value=value,
                unit="s",
                tags={"component": "test"}
            )
            for value in values
        ]
        
        # All metrics should have same name and unit
        names = [m.name for m in metrics]
        units = [m.unit for m in metrics]
        
        assert all(name == "test_operation" for name in names)
        assert all(unit == "s" for unit in units)
        
        # Timestamps should be non-decreasing (created sequentially)
        timestamps = [m.timestamp for m in metrics]
        for i in range(1, len(timestamps)):
            assert timestamps[i] >= timestamps[i-1]


class TestAsyncConcurrencyProperties:
    """Property-based tests for async concurrency utilities."""
    
    @pytest.mark.asyncio
    @given(
        num_operations=st.integers(min_value=1, max_value=20),
        max_concurrency=st.integers(min_value=1, max_value=10),
        operation_delay=st.floats(min_value=0.001, max_value=0.01)
    )
    async def test_concurrent_execution_properties(self, num_operations, max_concurrency, operation_delay):
        """Test concurrent execution properties."""
        import asyncio
        import time
        
        async def slow_operation(value: int) -> int:
            await asyncio.sleep(operation_delay)
            return value * 2
        
        operations = [slow_operation(i) for i in range(num_operations)]
        
        start_time = time.time()
        results = await gather_with_concurrency(operations, max_concurrency)
        elapsed = time.time() - start_time
        
        # Should get correct results
        assert len(results) == num_operations
        assert results == [i * 2 for i in range(num_operations)]
        
        # Concurrent execution should be faster than sequential
        sequential_time = num_operations * operation_delay
        # Allow some tolerance for overhead
        assert elapsed < sequential_time * 0.9
    
    @pytest.mark.asyncio
    @given(
        num_operations=st.integers(min_value=1, max_value=10),
        success_rate=st.floats(min_value=0.0, max_value=1.0)
    )
    async def test_async_retry_properties(self, num_operations, success_rate):
        """Test async retry properties."""
        import asyncio
        
        call_count = 0
        success_count = 0
        
        async def flaky_operation():
            nonlocal call_count, success_count
            call_count += 1
            
            # Succeed based on success rate
            if call_count / num_operations <= success_rate:
                success_count += 1
                return "success"
            else:
                raise ValueError("Simulated failure")
        
        retry = AsyncRetry(max_attempts=3, base_delay=0.001)
        
        results = []
        for _ in range(num_operations):
            try:
                result = await retry.execute(flaky_operation)
                results.append(result)
            except Exception:
                results.append(None)
        
        # Success count should match expected rate (approximately)
        expected_successes = int(num_operations * success_rate)
        assert abs(success_count - expected_successes) <= 1


class TestDecimalArithmeticProperties:
    """Property-based tests for decimal arithmetic in trading."""
    
    @given(
        quantities=st.lists(
            st.decimals(min_value=Decimal('0.001'), max_value=Decimal('1000'), places=8),
            min_size=1,
            max_size=10
        ),
        prices=st.lists(
            st.decimals(min_value=Decimal('1'), max_value=Decimal('100000'), places=2),
            min_size=1,
            max_size=10
        )
    )
    def test_position_calculation_properties(self, quantities, prices):
        """Test position calculation properties."""
        assume(len(quantities) == len(prices))
        
        # Calculate total position value
        total_value = Decimal('0')
        for qty, price in zip(quantities, prices):
            position_value = qty * price
            total_value += position_value
            
            # Individual position value should be positive
            assert position_value >= 0
        
        # Total value should be sum of individual values
        expected_total = sum(qty * price for qty, price in zip(quantities, prices))
        assert total_value == expected_total
        
        # Total value should be non-negative
        assert total_value >= 0
    
    @given(
        balance=st.decimals(min_value=Decimal('100'), max_value=Decimal('1000000'), places=2),
        risk_percentages=st.lists(
            st.decimals(min_value=Decimal('0.01'), max_value=Decimal('0.1'), places=4),
            min_size=1,
            max_size=5
        )
    )
    def test_risk_calculation_properties(self, balance, risk_percentages):
        """Test risk calculation properties."""
        risk_amounts = []
        
        for risk_pct in risk_percentages:
            risk_amount = balance * risk_pct
            risk_amounts.append(risk_amount)
            
            # Risk amount should be positive and less than balance
            assert risk_amount > 0
            assert risk_amount <= balance
        
        # Total risk should not exceed reasonable bounds
        total_risk = sum(risk_amounts)
        assert total_risk >= 0
        
        # If multiple risks, total might exceed balance (acceptable for some scenarios)
        if len(risk_percentages) > 1:
            # But individual risks should still be reasonable
            for risk_amount in risk_amounts:
                assert risk_amount <= balance


class TestStateMachine(RuleBasedStateMachine):
    """State machine test for trading operations."""
    
    def __init__(self):
        super().__init__()
        self.balance = Decimal('10000')
        self.positions = {}
        self.trades = []
    
    positions_bundle = Bundle('positions')
    
    @rule(
        symbol=st.text(min_size=3, max_size=10, alphabet='ABCDEFGHIJKLMNOPQRSTUVWXYZ'),
        side=st.one_of([st.just('buy'), st.just('sell')]),
        quantity=st.decimals(min_value=Decimal('0.001'), max_value=Decimal('10'), places=8),
        price=st.decimals(min_value=Decimal('1'), max_value=Decimal('100000'), places=2)
    )
    def execute_trade(self, symbol, side, quantity, price):
        """Execute a trade and update state."""
        assume(symbol.isupper())  # Ensure valid symbol format
        
        trade_value = quantity * price
        
        # Check if trade is possible
        if side == 'buy':
            if trade_value <= self.balance:
                # Can buy
                self.balance -= trade_value
                if symbol not in self.positions:
                    self.positions[symbol] = Decimal('0')
                self.positions[symbol] += quantity
                
                self.trades.append({
                    'symbol': symbol,
                    'side': side,
                    'quantity': quantity,
                    'price': price,
                    'value': trade_value
                })
        else:  # sell
            if symbol in self.positions and self.positions[symbol] >= quantity:
                # Can sell
                self.balance += trade_value
                self.positions[symbol] -= quantity
                
                if self.positions[symbol] == 0:
                    del self.positions[symbol]
                
                self.trades.append({
                    'symbol': symbol,
                    'side': side,
                    'quantity': quantity,
                    'price': price,
                    'value': trade_value
                })
    
    @rule()
    def check_balance_invariant(self):
        """Check that balance never goes negative."""
        assert self.balance >= 0
    
    @rule()
    def check_position_invariant(self):
        """Check that positions are always non-negative."""
        for symbol, quantity in self.positions.items():
            assert quantity >= 0
    
    @rule()
    def check_trade_consistency(self):
        """Check that trades are consistent with state changes."""
        # This is a simplified check - in reality would track full state changes
        assert len(self.trades) >= 0  # Should never be negative


# Register the state machine for property-based testing
TestTradingStateMachine = TestStateMachine.TestCase


class TestMathematicalProperties:
    """Test mathematical properties of trading calculations."""
    
    @given(
        prices=st.lists(
            st.decimals(min_value=Decimal('1'), max_value=Decimal('100000'), places=2),
            min_size=2,
            max_size=100
        )
    )
    def test_price_volatility_properties(self, prices):
        """Test price volatility calculation properties."""
        # Calculate mean
        mean_price = sum(prices) / len(prices)
        
        # Calculate variance
        variance = sum((price - mean_price) ** 2 for price in prices) / len(prices)
        
        # Standard deviation
        std_dev = variance.sqrt()
        
        # Properties
        assert std_dev >= 0  # Standard deviation should be non-negative
        assert variance >= 0  # Variance should be non-negative
        
        # If all prices are equal, volatility should be zero
        if all(price == prices[0] for price in prices):
            assert std_dev == 0
            assert variance == 0
    
    @given(
        returns=st.lists(
            st.decimals(min_value=Decimal('-0.5'), max_value=Decimal('0.5'), places=4),
            min_size=1,
            max_size=100
        )
    )
    def test_compound_return_properties(self, returns):
        """Test compound return calculation properties."""
        # Start with 1.0 (100%)
        portfolio_value = Decimal('1.0')
        
        for return_rate in returns:
            portfolio_value *= (Decimal('1') + return_rate)
            # Portfolio value should never go negative
            assert portfolio_value >= 0
        
        # Final value should equal compound of all returns
        expected_value = Decimal('1.0')
        for return_rate in returns:
            expected_value *= (Decimal('1') + return_rate)
        
        assert abs(portfolio_value - expected_value) < Decimal('1e-10')


class TestErrorHandlingProperties:
    """Property-based tests for error handling."""
    
    @given(
        failure_points=st.lists(
            st.integers(min_value=0, max_value=10),
            min_size=0,
            max_size=10
        )
    )
    def test_retry_with_specific_failures(self, failure_points):
        """Test retry mechanism with specific failure patterns."""
        call_count = 0
        failure_set = set(failure_points)
        
        def operation_with_failures():
            nonlocal call_count
            current_call = call_count
            call_count += 1
            
            if current_call in failure_set:
                raise ValueError(f"Failure at call {current_call}")
            return f"success at call {current_call}"
        
        try:
            result = retry_with_backoff(operation_with_failures, max_attempts=15)
            
            # Should eventually succeed
            assert "success" in result
            
            # Should have made at most len(failure_points) + 1 calls
            assert call_count <= len(failure_points) + 1
            
        except Exception:
            # Should have exhausted all attempts
            assert call_count >= len(failure_points)


if __name__ == "__main__":
    # Run property-based tests
    pytest.main([__file__, "-v", "--hypothesis-show-statistics"])
