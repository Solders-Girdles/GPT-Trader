"""
Equal weight portfolio allocator.
"""

from typing import Dict, Any, List
from core.interfaces import IPortfolioAllocator, ComponentConfig


class EqualWeightAllocator(IPortfolioAllocator):
    """
    Simple allocator that equally weights positions across signals.
    
    Features:
    - Equal allocation to all active signals
    - Respects risk constraints
    - Rebalancing threshold logic
    - Position limit enforcement
    """
    
    def __init__(self, config: ComponentConfig):
        """
        Initialize the allocator.
        
        Args:
            config: Component configuration
        """
        super().__init__(config)
        self.max_positions = config.config.get('max_positions', 5)
        self.min_allocation = config.config.get('min_allocation', 0.05)  # 5% minimum
        self.cash_reserve = config.config.get('cash_reserve', 0.05)  # Keep 5% cash
        
    def initialize(self) -> None:
        """Initialize the allocator."""
        pass
    
    def shutdown(self) -> None:
        """Cleanup and shutdown."""
        pass
    
    def allocate(
        self,
        signals: Dict[str, float],
        portfolio_value: float,
        current_positions: Dict[str, float],
        risk_constraints: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Allocate portfolio based on signals and constraints.
        
        Equal weight allocation with constraints:
        1. Only allocate to positive signals (buys)
        2. Limit number of positions
        3. Respect minimum allocation size
        4. Keep cash reserve
        
        Args:
            signals: Dict of symbol -> signal strength
            portfolio_value: Total portfolio value
            current_positions: Current position values by symbol
            risk_constraints: Risk limits from risk manager
            
        Returns:
            Dict of symbol -> target_allocation (fraction of portfolio)
        """
        allocations = {}
        
        # Filter for actionable buy signals
        buy_signals = {
            symbol: strength 
            for symbol, strength in signals.items() 
            if strength > 0
        }
        
        if not buy_signals:
            # No buy signals, maintain current or go to cash
            return allocations
        
        # Limit number of positions
        symbols_to_allocate = sorted(
            buy_signals.keys(),
            key=lambda s: buy_signals[s],
            reverse=True
        )[:self.max_positions]
        
        # Calculate available capital
        available_capital = portfolio_value * (1 - self.cash_reserve)
        
        # Apply risk constraints if provided
        if risk_constraints:
            max_position_size = risk_constraints.get('max_position_size', 1.0)
            available_capital = min(
                available_capital,
                portfolio_value * max_position_size * len(symbols_to_allocate)
            )
        
        # Equal weight allocation
        num_positions = len(symbols_to_allocate)
        if num_positions > 0:
            allocation_per_position = available_capital / portfolio_value / num_positions
            
            # Ensure minimum allocation
            if allocation_per_position < self.min_allocation:
                # Reduce number of positions to meet minimum
                max_positions_with_min = int(available_capital / portfolio_value / self.min_allocation)
                symbols_to_allocate = symbols_to_allocate[:max_positions_with_min]
                num_positions = len(symbols_to_allocate)
                
                if num_positions > 0:
                    allocation_per_position = available_capital / portfolio_value / num_positions
            
            # Create allocations
            for symbol in symbols_to_allocate:
                # Weight by signal strength (optional enhancement)
                # For now, pure equal weight
                allocations[symbol] = allocation_per_position
        
        return allocations
    
    def rebalance_required(
        self,
        current_positions: Dict[str, float],
        target_allocations: Dict[str, float],
        threshold: float = 0.05
    ) -> bool:
        """
        Check if rebalancing is needed.
        
        Rebalance if any position deviates from target by more than threshold.
        
        Args:
            current_positions: Current position values (absolute)
            target_allocations: Target allocations (fractions)
            threshold: Rebalancing threshold (5% default)
            
        Returns:
            True if rebalancing needed
        """
        # Calculate total portfolio value
        total_value = sum(current_positions.values())
        if total_value == 0:
            return bool(target_allocations)  # Rebalance if we have targets but no positions
        
        # Convert current positions to allocations
        current_allocations = {
            symbol: value / total_value
            for symbol, value in current_positions.items()
        }
        
        # Check each position
        all_symbols = set(current_allocations.keys()) | set(target_allocations.keys())
        
        for symbol in all_symbols:
            current = current_allocations.get(symbol, 0)
            target = target_allocations.get(symbol, 0)
            
            # Check deviation
            deviation = abs(current - target)
            if deviation > threshold:
                return True
        
        return False
    
    def get_allocation_metrics(self) -> Dict[str, Any]:
        """
        Get allocation performance metrics.
        
        Returns:
            Dict of metrics
        """
        return {
            'allocator_type': 'equal_weight',
            'max_positions': self.max_positions,
            'min_allocation': self.min_allocation,
            'cash_reserve': self.cash_reserve,
            'status': 'active'
        }