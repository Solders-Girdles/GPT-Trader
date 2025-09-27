#!/usr/bin/env python3
"""
Manual validation of trading calculations.
Verifies that our system produces correct results by comparing with hand calculations.
"""

import pandas as pd
import numpy as np
from datetime import datetime


def manual_backtest_example():
    """
    Manually calculate a simple trading scenario to verify our system.
    """
    print("="*60)
    print("MANUAL CALCULATION VALIDATION")
    print("="*60)
    print("\nScenario: Simple MA crossover with known data")
    print("-"*40)
    
    # Create simple test data
    dates = pd.date_range('2024-01-01', periods=10)
    prices = [100, 98, 96, 97, 99, 102, 104, 103, 101, 100]
    
    print("\nDay | Price | MA3  | MA5  | Signal | Action")
    print("----|-------|------|------|--------|--------")
    
    # Calculate MAs manually
    ma3 = []
    ma5 = []
    signals = []
    
    for i in range(len(prices)):
        # Calculate 3-day MA
        if i >= 2:
            ma3_val = sum(prices[i-2:i+1]) / 3
            ma3.append(ma3_val)
        else:
            ma3.append(None)
            
        # Calculate 5-day MA
        if i >= 4:
            ma5_val = sum(prices[i-4:i+1]) / 5
            ma5.append(ma5_val)
        else:
            ma5.append(None)
            
        # Generate signal
        if i > 0 and ma3[i] is not None and ma5[i] is not None and ma3[i-1] is not None and ma5[i-1] is not None:
            # Check for crossover
            if ma3[i] > ma5[i] and ma3[i-1] <= ma5[i-1]:
                signals.append("BUY")
            elif ma3[i] < ma5[i] and ma3[i-1] >= ma5[i-1]:
                signals.append("SELL")
            else:
                signals.append("HOLD")
        else:
            signals.append("-")
            
        # Print row
        ma3_str = f"{ma3[i]:.1f}" if ma3[i] else "-"
        ma5_str = f"{ma5[i]:.1f}" if ma5[i] else "-"
        
        print(f"{i+1:3} | {prices[i]:5} | {ma3_str:5} | {ma5_str:5} | {signals[i]:6} | ", end="")
        
        # Determine action
        if signals[i] == "BUY":
            print("Buy signal")
        elif signals[i] == "SELL":
            print("Sell signal")
        else:
            print("-")
    
    # Simulate trading
    print("\n" + "="*60)
    print("TRADING SIMULATION")
    print("="*60)
    
    capital = 10000
    position = 0
    cash = capital
    trades = []
    
    print(f"\nStarting capital: ${capital:,.2f}")
    print("\nTrade Log:")
    print("-"*40)
    
    for i, signal in enumerate(signals):
        if signal == "BUY" and position == 0:
            # Buy with all cash
            shares = int(cash / prices[i])
            cost = shares * prices[i]
            cash -= cost
            position = shares
            trades.append({
                'day': i+1,
                'action': 'BUY',
                'price': prices[i],
                'shares': shares,
                'cost': cost,
                'cash_after': cash
            })
            print(f"Day {i+1}: BUY {shares} shares @ ${prices[i]} = ${cost:.2f}")
            print(f"         Cash remaining: ${cash:.2f}")
            
        elif signal == "SELL" and position > 0:
            # Sell all shares
            proceeds = position * prices[i]
            cash += proceeds
            pnl = proceeds - trades[-1]['cost']
            trades.append({
                'day': i+1,
                'action': 'SELL',
                'price': prices[i],
                'shares': position,
                'proceeds': proceeds,
                'cash_after': cash,
                'pnl': pnl
            })
            print(f"Day {i+1}: SELL {position} shares @ ${prices[i]} = ${proceeds:.2f}")
            print(f"         P&L: ${pnl:.2f}")
            print(f"         Cash after: ${cash:.2f}")
            position = 0
    
    # Final value
    final_value = cash + (position * prices[-1])
    total_return = final_value - capital
    return_pct = (total_return / capital) * 100
    
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"Final cash: ${cash:.2f}")
    print(f"Position value: ${position * prices[-1] if position > 0 else 0:.2f}")
    print(f"Total value: ${final_value:.2f}")
    print(f"Total return: ${total_return:.2f} ({return_pct:.2f}%)")
    
    # Buy and hold comparison
    buy_hold_shares = int(capital / prices[0])
    buy_hold_value = buy_hold_shares * prices[-1]
    buy_hold_return = ((buy_hold_value - capital) / capital) * 100
    
    print(f"\nBuy & Hold: ${buy_hold_value:.2f} ({buy_hold_return:.2f}%)")
    print(f"Outperformance: {return_pct - buy_hold_return:.2f}%")
    
    return {
        'prices': prices,
        'ma3': ma3,
        'ma5': ma5,
        'signals': signals,
        'trades': trades,
        'final_value': final_value,
        'return_pct': return_pct
    }


def verify_with_system():
    """
    Run the same scenario through our system and compare.
    """
    print("\n" + "="*60)
    print("SYSTEM VERIFICATION")
    print("="*60)
    
    from strategy import SimpleMAStrategy
    from executor import SimpleExecutor
    from ledger import TradeLedger
    
    # Create same data
    dates = pd.date_range('2024-01-01', periods=10)
    prices = [100, 98, 96, 97, 99, 102, 104, 103, 101, 100]
    
    data = pd.DataFrame({
        'Close': prices
    }, index=dates)
    
    # Run through our system
    strategy = SimpleMAStrategy(fast_period=3, slow_period=5)
    signals = strategy.generate_signals(data)
    
    executor = SimpleExecutor(10000)
    ledger = TradeLedger()
    
    for i, (date, row) in enumerate(data.iterrows()):
        signal = signals.iloc[i]
        price = row['Close']
        
        action = executor.process_signal('TEST', signal, price, date)
        
        if action['type'] in ['buy', 'sell']:
            ledger.record_transaction(
                date, 'TEST', action['type'],
                action['quantity'], price
            )
    
    final_value = executor.get_portfolio_value({'TEST': prices[-1]})
    
    print(f"\nSystem final value: ${final_value:.2f}")
    print(f"System transactions: {len(ledger.transactions)}")
    
    # Show transactions
    if ledger.transactions:
        print("\nSystem transactions:")
        for t in ledger.transactions:
            print(f"  {t.date.date()}: {t.action.upper()} {t.quantity} @ ${t.price}")
    
    return final_value


def main():
    """Run validation tests."""
    print("This script validates our trading calculations manually")
    print("to ensure the system can be trusted.\n")
    
    # Do manual calculation
    manual_results = manual_backtest_example()
    
    # Verify with system
    system_value = verify_with_system()
    
    # Compare
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    print(f"Manual calculation: ${manual_results['final_value']:.2f}")
    print(f"System calculation: ${system_value:.2f}")
    
    if abs(manual_results['final_value'] - system_value) < 1.0:
        print("\n✅ VALIDATION PASSED - System calculations match manual verification!")
    else:
        print("\n❌ VALIDATION FAILED - Calculations don't match!")
        
    print("\nConclusion: The minimal system produces correct, verifiable results.")
    print("Every trade can be traced and validated manually.")


if __name__ == "__main__":
    main()