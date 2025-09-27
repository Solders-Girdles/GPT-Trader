"""
Behavioral validators for testing trading system outcomes.

These validators check that actual behavior matches expected outcomes.
"""

from decimal import Decimal
from typing import Dict, List, Optional, Any


def validate_pnl_calculation(
    trades: List[Dict[str, Any]],
    expected_pnl: Decimal,
    tolerance: Decimal = Decimal('0.01')
) -> tuple[bool, str]:
    """
    Validate that P&L calculation matches expected value.
    
    Args:
        trades: List of trade executions
        expected_pnl: Expected P&L value
        tolerance: Acceptable difference (for rounding)
        
    Returns:
        (success, message) tuple
    """
    # Calculate actual P&L from trades
    position = Decimal('0')
    cost_basis = Decimal('0')
    realized_pnl = Decimal('0')
    
    for trade in trades:
        side = trade['side']
        qty = Decimal(str(trade['quantity']))
        price = Decimal(str(trade['price']))
        
        if position == 0:
            # Opening position
            position = qty if side == 'buy' else -qty
            cost_basis = price
        elif (position > 0 and side == 'sell') or (position < 0 and side == 'buy'):
            # Closing or reducing position
            close_qty = min(qty, abs(position))
            
            if position > 0:  # Long position
                pnl = (price - cost_basis) * close_qty
            else:  # Short position
                pnl = (cost_basis - price) * close_qty
            
            realized_pnl += pnl
            position = position - close_qty if position > 0 else position + close_qty
            
            # Handle position flip
            if qty > abs(position):
                remaining = qty - abs(position)
                position = -remaining if side == 'sell' else remaining
                cost_basis = price
        else:
            # Adding to position
            # Update weighted average cost
            total_qty = abs(position) + qty
            cost_basis = ((abs(position) * cost_basis) + (qty * price)) / total_qty
            position = position + qty if side == 'buy' else position - qty
    
    # Check if P&L matches expected
    diff = abs(realized_pnl - expected_pnl)
    if diff <= tolerance:
        return True, f"P&L validated: {realized_pnl} ≈ {expected_pnl}"
    else:
        return False, f"P&L mismatch: actual={realized_pnl}, expected={expected_pnl}, diff={diff}"


def validate_position_state(
    position: Dict[str, Any],
    expected_state: Dict[str, Any]
) -> tuple[bool, str]:
    """
    Validate position state matches expected values.
    
    Args:
        position: Actual position state
        expected_state: Expected position values
        
    Returns:
        (success, message) tuple
    """
    errors = []
    
    # Check quantity
    if 'quantity' in expected_state:
        expected_qty = Decimal(str(expected_state['quantity']))
        actual_qty = Decimal(str(position.get('quantity', 0)))
        if abs(actual_qty - expected_qty) > Decimal('0.0001'):
            errors.append(f"Quantity mismatch: {actual_qty} != {expected_qty}")
    
    # Check side
    if 'side' in expected_state:
        if position.get('side') != expected_state['side']:
            errors.append(f"Side mismatch: {position.get('side')} != {expected_state['side']}")
    
    # Check average entry price
    if 'avg_entry_price' in expected_state:
        expected_price = Decimal(str(expected_state['avg_entry_price']))
        actual_price = Decimal(str(position.get('avg_entry_price', 0)))
        if abs(actual_price - expected_price) > Decimal('0.01'):
            errors.append(f"Entry price mismatch: {actual_price} != {expected_price}")
    
    # Check realized P&L
    if 'realized_pnl' in expected_state:
        expected_pnl = Decimal(str(expected_state['realized_pnl']))
        actual_pnl = Decimal(str(position.get('realized_pnl', 0)))
        if abs(actual_pnl - expected_pnl) > Decimal('0.01'):
            errors.append(f"Realized P&L mismatch: {actual_pnl} != {expected_pnl}")
    
    if errors:
        return False, "; ".join(errors)
    return True, "Position state validated"


def validate_risk_limits(
    trades: List[Dict[str, Any]],
    risk_config: Dict[str, Any]
) -> tuple[bool, str]:
    """
    Validate that risk limits are enforced.
    
    Args:
        trades: List of executed trades
        risk_config: Risk configuration with limits
        
    Returns:
        (success, message) tuple
    """
    violations = []
    
    # Check position size limits
    max_position = Decimal(str(risk_config.get('max_position_size', float('inf'))))
    current_position = Decimal('0')
    
    for trade in trades:
        qty = Decimal(str(trade['quantity']))
        side = trade['side']
        
        # Update position
        if side == 'buy':
            current_position += qty
        else:
            current_position -= qty
        
        # Check if position exceeds limit
        if abs(current_position) > max_position:
            violations.append(
                f"Position size violation: {abs(current_position)} > {max_position}"
            )
    
    # Check daily loss limit
    if 'max_daily_loss' in risk_config:
        max_loss = Decimal(str(risk_config['max_daily_loss']))
        daily_pnl = Decimal('0')
        
        # Calculate daily P&L
        for i, trade in enumerate(trades):
            if trade.get('pnl'):
                daily_pnl += Decimal(str(trade['pnl']))
                
                if daily_pnl < -max_loss:
                    violations.append(
                        f"Daily loss limit violated: {daily_pnl} < -{max_loss}"
                    )
    
    # Check leverage limits
    if 'max_leverage' in risk_config:
        max_leverage = Decimal(str(risk_config['max_leverage']))
        
        for trade in trades:
            leverage = Decimal(str(trade.get('leverage', 1)))
            if leverage > max_leverage:
                violations.append(
                    f"Leverage limit violated: {leverage} > {max_leverage}"
                )
    
    if violations:
        return False, "; ".join(violations)
    return True, "All risk limits respected"


def validate_order_execution(
    order: Dict[str, Any],
    product_spec: Dict[str, Any],
    expected_outcome: Dict[str, Any]
) -> tuple[bool, str]:
    """
    Validate order execution matches product specifications.
    
    Args:
        order: Executed order details
        product_spec: Product specifications (min size, tick size, etc.)
        expected_outcome: Expected execution outcome
        
    Returns:
        (success, message) tuple
    """
    errors = []
    
    # Extract specs
    min_size = Decimal(str(product_spec.get('min_size', 0)))
    step_size = Decimal(str(product_spec.get('step_size', 0)))
    tick_size = Decimal(str(product_spec.get('tick_size', 0)))
    min_notional = Decimal(str(product_spec.get('min_notional', 0)))
    
    # Extract order details
    qty = Decimal(str(order['quantity']))
    price = Decimal(str(order.get('price', 0)))
    
    # Validate quantity
    if qty < min_size:
        errors.append(f"Quantity below minimum: {qty} < {min_size}")
    
    if step_size > 0:
        remainder = qty % step_size
        if remainder > Decimal('0.0000001'):
            errors.append(f"Quantity not aligned to step size: {qty} % {step_size} = {remainder}")
    
    # Validate price (for limit orders)
    if price > 0 and tick_size > 0:
        remainder = price % tick_size
        if remainder > Decimal('0.0000001'):
            errors.append(f"Price not aligned to tick size: {price} % {tick_size} = {remainder}")
    
    # Validate notional
    if price > 0:
        notional = qty * price
        if notional < min_notional:
            errors.append(f"Notional below minimum: {notional} < {min_notional}")
    
    # Validate expected outcome
    if expected_outcome:
        if 'adjusted_quantity' in expected_outcome:
            expected_qty = Decimal(str(expected_outcome['adjusted_quantity']))
            if abs(qty - expected_qty) > Decimal('0.0001'):
                errors.append(f"Quantity adjustment failed: {qty} != {expected_qty}")
        
        if 'adjusted_price' in expected_outcome:
            expected_price = Decimal(str(expected_outcome['adjusted_price']))
            if abs(price - expected_price) > Decimal('0.01'):
                errors.append(f"Price adjustment failed: {price} != {expected_price}")
        
        if 'rejected' in expected_outcome and expected_outcome['rejected']:
            if order.get('status') != 'rejected':
                errors.append("Order should have been rejected but wasn't")
    
    if errors:
        return False, "; ".join(errors)
    return True, "Order execution validated"


def validate_funding_accrual(
    position: Dict[str, Any],
    funding_scenario: Dict[str, Any],
    current_time: Any
) -> tuple[bool, str]:
    """
    Validate funding payment accrual for perpetuals.
    
    Args:
        position: Current position state
        funding_scenario: Funding scenario details
        current_time: Current timestamp
        
    Returns:
        (success, message) tuple
    """
    expected_payments = Decimal('0')
    payment_per_period = Decimal(str(funding_scenario['payment_per_period']))
    
    # Count how many funding periods have passed
    periods_passed = 0
    for funding_time in funding_scenario['funding_times']:
        if current_time >= funding_time:
            periods_passed += 1
    
    expected_payments = payment_per_period * periods_passed
    
    # Get actual funding paid from position
    actual_funding = Decimal(str(position.get('funding_paid', 0)))
    
    # Check if funding matches (note: sign convention varies)
    diff = abs(actual_funding - expected_payments)
    if diff <= Decimal('0.01'):
        return True, f"Funding validated: {actual_funding} ≈ {expected_payments}"
    else:
        return False, f"Funding mismatch: actual={actual_funding}, expected={expected_payments}"