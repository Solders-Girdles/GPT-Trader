"""
Trade execution simulation for backtesting.
"""

from typing import List, Dict, Tuple
import pandas as pd
import logging
import numpy as np

from ...errors import ExecutionError, ValidationError, InsufficientFundsError
from ...validation import (
    validate_inputs, DataFrameValidator, SeriesValidator,
    PositiveNumberValidator, PercentageValidator, RangeValidator
)
from ...config import get_config

logger = logging.getLogger(__name__)


@validate_inputs(
    signals=SeriesValidator(),  # Signals must be a pandas Series
    data=DataFrameValidator(
        required_columns=['open', 'high', 'low', 'close', 'volume'],
        min_rows=1
    ),
    initial_capital=PositiveNumberValidator(),
    commission=PercentageValidator(as_decimal=True),
    slippage=PercentageValidator(as_decimal=True),
    position_size=RangeValidator(min_value=0.01, max_value=1.0)
)
def simulate_trades(
    signals: pd.Series,
    data: pd.DataFrame,
    initial_capital: float,
    commission: float = 0.001,
    slippage: float = 0.0005,
    position_size: float = 0.95
) -> Tuple[List[Dict], pd.Series, pd.Series]:
    """
    Simulate trade execution based on signals.
    
    Args:
        signals: Trading signals (1=buy, -1=sell, 0=hold)
        data: Market data with OHLCV columns
        initial_capital: Starting capital
        commission: Commission rate (as decimal)
        slippage: Slippage rate (as decimal)
        position_size: Fraction of capital to use per trade
        
    Returns:
        Tuple of (trades, equity_curve, returns)
        
    Raises:
        ExecutionError: If trade execution fails
        ValidationError: If inputs are invalid
        InsufficientFundsError: If not enough capital for trades
    """
    logger.info(
        f"Starting trade simulation with ${initial_capital:,.2f} initial capital",
        extra={
            'initial_capital': initial_capital,
            'commission': commission,
            'slippage': slippage,
            'position_size': position_size,
            'signals_count': len(signals),
            'data_rows': len(data)
        }
    )
    # Validate inputs before processing
    _validate_execution_inputs(signals, data, initial_capital, commission, slippage, position_size)
    
    # Load configuration for execution settings
    config = get_config('backtest')
    
    # Initialize trading state
    trades = []
    cash = float(initial_capital)  # Ensure float
    position_qty = 0
    equity_curve = []
    daily_returns = []
    trade_id = 0
    
    # Track execution statistics
    total_buy_signals = (signals == 1).sum()
    total_sell_signals = (signals == -1).sum()
    executed_buys = 0
    executed_sells = 0
    
    logger.debug(
        f"Processing {len(signals)} signals: {total_buy_signals} buys, {total_sell_signals} sells",
        extra={
            'total_signals': len(signals),
            'buy_signals': total_buy_signals,
            'sell_signals': total_sell_signals
        }
    )
    
    try:
        for i, (date, signal) in enumerate(signals.items()):
            # Validate signal value
            if not isinstance(signal, (int, float, np.integer, np.floating)):
                logger.warning(f"Invalid signal type at {date}: {type(signal)}")
                signal = 0
            
            # Get current price with validation
            try:
                current_price = float(data.loc[date, 'close'])
                if current_price <= 0 or np.isnan(current_price):
                    raise ExecutionError(
                        f"Invalid price at {date}: {current_price}",
                        context={'date': str(date), 'price': current_price}
                    )
            except KeyError:
                raise ExecutionError(
                    f"Missing data for date {date}",
                    context={'date': str(date), 'available_dates': len(data)}
                )
            
            # Calculate current equity with safety checks
            position_value = position_qty * current_price if position_qty > 0 else 0
            current_equity = cash + position_value
            
            if current_equity <= 0:
                logger.warning(f"Negative equity detected at {date}: ${current_equity:,.2f}")
            
            equity_curve.append(current_equity)
            
            # Calculate daily return with safety checks
            if i > 0:
                prev_equity = equity_curve[i-1]
                if prev_equity > 0:
                    daily_return = (current_equity - prev_equity) / prev_equity
                else:
                    daily_return = 0.0
                daily_returns.append(daily_return)
            else:
                daily_returns.append(0.0)
            
            # Process buy signal
            if signal == 1 and position_qty == 0:
                try:
                    trade_result = _execute_buy_order(
                        date, current_price, cash, position_size, 
                        commission, slippage, trade_id
                    )
                    
                    if trade_result:
                        trade, new_cash, new_qty = trade_result
                        trades.append(trade)
                        cash = new_cash
                        position_qty = new_qty
                        trade_id += 1
                        executed_buys += 1
                        
                        logger.debug(
                            f"Executed buy: {new_qty} shares at ${trade['price']:.2f}",
                            extra={
                                'date': str(date),
                                'price': trade['price'],
                                'quantity': new_qty,
                                'remaining_cash': new_cash
                            }
                        )
                        
                except Exception as e:
                    logger.warning(f"Buy order failed at {date}: {e}")
            
            # Process sell signal
            elif signal == -1 and position_qty > 0:
                try:
                    trade_result = _execute_sell_order(
                        date, current_price, position_qty, 
                        commission, slippage, trade_id
                    )
                    
                    if trade_result:
                        trade, proceeds = trade_result
                        trades.append(trade)
                        cash += proceeds
                        trade_id += 1
                        executed_sells += 1
                        
                        logger.debug(
                            f"Executed sell: {position_qty} shares at ${trade['price']:.2f}",
                            extra={
                                'date': str(date),
                                'price': trade['price'],
                                'quantity': position_qty,
                                'proceeds': proceeds,
                                'new_cash': cash
                            }
                        )
                        
                        position_qty = 0
                        
                except Exception as e:
                    logger.warning(f"Sell order failed at {date}: {e}")
                    
    except Exception as e:
        raise ExecutionError(
            f"Trade execution failed during simulation",
            context={
                'processed_signals': i if 'i' in locals() else 0,
                'total_signals': len(signals),
                'cash': cash,
                'position_qty': position_qty,
                'error': str(e)
            }
        ) from e
    
    # Close any remaining position at the end
    try:
        if position_qty > 0:
            final_date = data.index[-1]
            final_price = float(data.iloc[-1]['close'])
            
            if final_price > 0:
                trade_result = _execute_sell_order(
                    final_date, final_price, position_qty, 
                    commission, slippage, trade_id, is_final=True
                )
                
                if trade_result:
                    trade, proceeds = trade_result
                    trades.append(trade)
                    cash += proceeds
                    executed_sells += 1
                    
                    logger.info(
                        f"Closed final position: {position_qty} shares at ${final_price:.2f}",
                        extra={
                            'date': str(final_date),
                            'price': final_price,
                            'quantity': position_qty,
                            'proceeds': proceeds
                        }
                    )
            else:
                logger.warning(f"Invalid final price for position closure: {final_price}")
                
    except Exception as e:
        logger.warning(f"Failed to close final position: {e}")
    
    # Create result series with validation
    try:
        equity_series = pd.Series(equity_curve, index=data.index, name='equity')
        returns_series = pd.Series(daily_returns, index=data.index, name='returns')
    except Exception as e:
        raise ExecutionError(
            "Failed to create result series",
            context={
                'equity_points': len(equity_curve),
                'returns_points': len(daily_returns),
                'data_index_length': len(data.index),
                'error': str(e)
            }
        ) from e
    
    # Log execution summary
    logger.info(
        f"Trade execution completed: {len(trades)} trades, ${equity_series.iloc[-1]:,.2f} final equity",
        extra={
            'total_trades': len(trades),
            'executed_buys': executed_buys,
            'executed_sells': executed_sells,
            'buy_signal_fill_rate': executed_buys / total_buy_signals if total_buy_signals > 0 else 0,
            'sell_signal_fill_rate': executed_sells / total_sell_signals if total_sell_signals > 0 else 0,
            'final_equity': equity_series.iloc[-1],
            'total_return': ((equity_series.iloc[-1] - initial_capital) / initial_capital) * 100
        }
    )
    
    return trades, equity_series, returns_series


def _validate_execution_inputs(
    signals: pd.Series,
    data: pd.DataFrame,
    initial_capital: float,
    commission: float,
    slippage: float,
    position_size: float
) -> None:
    """
    Validate inputs for trade execution.
    
    Args:
        signals: Trading signals
        data: Market data
        initial_capital: Starting capital
        commission: Commission rate
        slippage: Slippage rate
        position_size: Position size fraction
        
    Raises:
        ValidationError: If validation fails
    """
    # Check signals and data alignment
    if len(signals) != len(data):
        raise ValidationError(
            "Signals and data must have same length",
            field="signals_data_alignment",
            value=f"signals: {len(signals)}, data: {len(data)}"
        )
    
    # Check signal values
    unique_signals = set(signals.dropna().unique())
    valid_signals = {-1, 0, 1}
    invalid_signals = unique_signals - valid_signals
    if invalid_signals:
        raise ValidationError(
            f"Invalid signal values: {invalid_signals}",
            field="signal_values",
            value=list(unique_signals)
        )
    
    # Check data quality
    if data['close'].isna().any():
        raise ValidationError(
            "Close prices contain NaN values",
            field="close_prices"
        )
    
    if (data['close'] <= 0).any():
        raise ValidationError(
            "Close prices contain non-positive values",
            field="close_prices",
            value=f"min_price: {data['close'].min()}"
        )


def _execute_buy_order(
    date,
    current_price: float,
    cash: float,
    position_size: float,
    commission: float,
    slippage: float,
    trade_id: int
) -> tuple:
    """
    Execute a buy order with validation.
    
    Args:
        date: Trade date
        current_price: Current market price
        cash: Available cash
        position_size: Position size fraction
        commission: Commission rate
        slippage: Slippage rate
        trade_id: Trade identifier
        
    Returns:
        Tuple of (trade_dict, new_cash, position_quantity) or None if failed
        
    Raises:
        ExecutionError: If execution fails
        InsufficientFundsError: If not enough cash
    """
    try:
        # Calculate buy price with slippage
        buy_price = current_price * (1 + slippage)
        
        # Calculate maximum quantity we can afford
        available_cash = cash * position_size
        gross_cost_per_share = buy_price
        net_cost_per_share = buy_price * (1 + commission)
        
        max_qty = int(available_cash / net_cost_per_share)
        
        if max_qty <= 0:
            raise InsufficientFundsError(
                f"Insufficient funds for buy order",
                required=net_cost_per_share,
                available=available_cash,
                context={
                    'date': str(date),
                    'buy_price': buy_price,
                    'available_cash': available_cash,
                    'cost_per_share': net_cost_per_share
                }
            )
        
        # Calculate actual costs
        gross_cost = max_qty * buy_price
        commission_cost = gross_cost * commission
        total_cost = gross_cost + commission_cost
        
        # Validate we have enough cash
        if total_cost > cash:
            raise InsufficientFundsError(
                f"Total cost exceeds available cash",
                required=total_cost,
                available=cash,
                context={
                    'date': str(date),
                    'quantity': max_qty,
                    'gross_cost': gross_cost,
                    'commission': commission_cost
                }
            )
        
        # Create trade record
        trade = {
            'id': trade_id,
            'date': date,
            'side': 'buy',
            'price': buy_price,
            'quantity': max_qty,
            'commission': commission_cost,
            'gross_amount': gross_cost,
            'net_amount': total_cost
        }
        
        new_cash = cash - total_cost
        
        return trade, new_cash, max_qty
        
    except (InsufficientFundsError, ExecutionError):
        raise
    except Exception as e:
        raise ExecutionError(
            f"Buy order execution failed",
            context={
                'date': str(date),
                'price': current_price,
                'cash': cash,
                'error': str(e)
            }
        ) from e


def _execute_sell_order(
    date,
    current_price: float,
    position_qty: int,
    commission: float,
    slippage: float,
    trade_id: int,
    is_final: bool = False
) -> tuple:
    """
    Execute a sell order with validation.
    
    Args:
        date: Trade date
        current_price: Current market price
        position_qty: Position quantity to sell
        commission: Commission rate
        slippage: Slippage rate
        trade_id: Trade identifier
        is_final: Whether this is a final position closure
        
    Returns:
        Tuple of (trade_dict, proceeds) or None if failed
        
    Raises:
        ExecutionError: If execution fails
    """
    try:
        if position_qty <= 0:
            raise ExecutionError(
                f"Invalid position quantity for sell: {position_qty}",
                context={'date': str(date), 'quantity': position_qty}
            )
        
        # Calculate sell price with slippage
        sell_price = current_price * (1 - slippage)
        
        if sell_price <= 0:
            raise ExecutionError(
                f"Invalid sell price: {sell_price}",
                context={
                    'date': str(date),
                    'current_price': current_price,
                    'slippage': slippage
                }
            )
        
        # Calculate proceeds
        gross_proceeds = position_qty * sell_price
        commission_cost = gross_proceeds * commission
        net_proceeds = gross_proceeds - commission_cost
        
        # Create trade record
        trade = {
            'id': trade_id,
            'date': date,
            'side': 'sell',
            'price': sell_price,
            'quantity': position_qty,
            'commission': commission_cost,
            'gross_amount': gross_proceeds,
            'net_amount': net_proceeds,
            'is_final_closure': is_final
        }
        
        return trade, net_proceeds
        
    except ExecutionError:
        raise
    except Exception as e:
        raise ExecutionError(
            f"Sell order execution failed",
            context={
                'date': str(date),
                'price': current_price,
                'quantity': position_qty,
                'error': str(e)
            }
        ) from e


def get_execution_statistics(trades: List[Dict]) -> Dict:
    """
    Calculate execution statistics from trades.
    
    Args:
        trades: List of trade dictionaries
        
    Returns:
        Dictionary with execution statistics
    """
    try:
        if not trades:
            return {
                'total_trades': 0,
                'buy_trades': 0,
                'sell_trades': 0,
                'total_commission': 0.0,
                'total_slippage_cost': 0.0,
                'avg_trade_size': 0.0
            }
        
        buy_trades = [t for t in trades if t['side'] == 'buy']
        sell_trades = [t for t in trades if t['side'] == 'sell']
        
        total_commission = sum(t.get('commission', 0) for t in trades)
        total_volume = sum(t['quantity'] * t['price'] for t in trades)
        avg_trade_size = total_volume / len(trades) if len(trades) > 0 else 0
        
        return {
            'total_trades': len(trades),
            'buy_trades': len(buy_trades),
            'sell_trades': len(sell_trades),
            'total_commission': total_commission,
            'total_volume': total_volume,
            'avg_trade_size': avg_trade_size,
            'avg_quantity_per_trade': sum(t['quantity'] for t in trades) / len(trades),
            'avg_price': sum(t['price'] for t in trades) / len(trades)
        }
        
    except Exception as e:
        logger.warning(f"Failed to calculate execution statistics: {e}")
        return {
            'total_trades': len(trades) if trades else 0,
            'error': str(e)
        }
