# 🧠 GPT-Trader Trading Logic Deep Dive

## Core Question: How Does GPT-Trader Decide to Trade?

This document traces the exact decision-making process from market data to executed trades, examining the logic at each step.

## 📈 The Trading Decision Pipeline

```
Market Data → Indicators → Signal → Allocation → Risk Check → Execution
     ↓            ↓          ↓           ↓            ↓           ↓
   OHLCV      MA, ATR     Buy/Sell   Position     Limits      Trade
             Donchian      Score      Sizing     Applied     Order
```

## 1️⃣ Signal Generation Logic

### DemoMAStrategy Example
```python
def generate_signals(self, df):
    # Calculate indicators
    sma_fast = df["close"].rolling(10).mean()  # 10-day MA
    sma_slow = df["close"].rolling(20).mean()  # 20-day MA
    
    # Generate signal
    signal = np.where(
        sma_fast > sma_slow,  # Condition
        1.0,                  # Buy signal
        0.0                   # No signal
    )
    
    # Risk levels (NEW - added for tests)
    atr = calculate_atr(df, period=14)
    stop_loss = df["close"] - (2 * atr)    # 2 ATR below
    take_profit = df["close"] + (3 * atr)  # 3 ATR above
```

**Decision Points:**
- **Entry**: Fast MA crosses above Slow MA → BUY
- **Exit**: Fast MA crosses below Slow MA → SELL
- **Stop Loss**: Price drops 2 ATR from entry
- **Take Profit**: Price rises 3 ATR from entry

### TrendBreakoutStrategy Logic
```python
def generate_signals(self, df):
    # Donchian Channel
    high_20 = df["high"].rolling(20).max()  # 20-day high
    low_20 = df["low"].rolling(20).min()    # 20-day low
    
    # Breakout signal
    signal = np.where(
        df["close"] > high_20.shift(1),  # Break above previous high
        1.0,                              # Buy signal
        0.0                               # No signal
    )
```

**Decision Points:**
- **Entry**: Price breaks above 20-day high
- **Exit**: Price falls below 20-day low
- **Strength**: How far above breakout level

## 2️⃣ Position Sizing Mathematics

### The Kelly Criterion Inspiration
```python
def position_size(equity, atr, price, rules):
    # Dynamic risk based on account size
    if equity < 1000:
        risk_pct = 0.02  # 2% for micro accounts
    elif equity < 5000:
        risk_pct = 0.01  # 1% for small accounts
    elif equity < 25000:
        risk_pct = 0.0075  # 0.75% for medium
    else:
        risk_pct = 0.005  # 0.5% for standard
    
    # Calculate position
    risk_dollars = equity * risk_pct
    stop_distance = 2 * atr  # Stop loss distance
    
    shares = floor(risk_dollars / stop_distance)
    
    # Example:
    # Equity: $10,000
    # Risk: 0.75% = $75
    # ATR: $2, Stop: 2 * $2 = $4
    # Shares: $75 / $4 = 18 shares
```

### Why This Formula?
- **Risk-based**: Limits loss per trade to fixed percentage
- **Volatility-adjusted**: Uses ATR to scale for market conditions
- **Account-scaled**: Smaller accounts take more risk to be viable

## 3️⃣ Allocation Selection Process

### How GPT-Trader Chooses Which Signals to Take

```python
def allocate_signals(signals, equity, rules):
    candidates = []
    
    # Step 1: Collect all active signals
    for symbol, data in signals.items():
        if signal > 0:  # Has buy signal
            strength = calculate_strength(data)
            position_size = calculate_size(equity, atr, price)
            candidates.append((symbol, strength, position_size))
    
    # Step 2: Rank by strength
    candidates.sort(key=lambda x: x[1], reverse=True)
    
    # Step 3: Select top N
    selected = candidates[:rules.max_positions]  # Default: 10
    
    return {sym: size for sym, _, size in selected}
```

### Strength Calculation Methods

**Method 1: Breakout Strength**
```python
strength = (current_price - breakout_level) / current_price
# Higher = stronger breakout
```

**Method 2: Signal/ATR Ratio**
```python
strength = signal_value / atr
# Higher signal with lower volatility = better
```

**Method 3: Momentum**
```python
strength = (price_now - price_20_days_ago) / price_20_days_ago
# Higher momentum = stronger trend
```

## 4️⃣ Risk Management Decisions

### Multi-Layer Risk Control

```python
def validate_allocations(allocations, prices, portfolio_value):
    # Layer 1: Position Size Limits
    for symbol, shares in allocations.items():
        position_value = shares * prices[symbol]
        position_pct = position_value / portfolio_value
        
        max_allowed = get_dynamic_limit(portfolio_value)
        # $1K portfolio: 50% max per position
        # $10K portfolio: 15% max per position
        # $100K portfolio: 10% max per position
        
        if position_pct > max_allowed:
            shares = scale_down_position(shares, max_allowed)
    
    # Layer 2: Total Exposure
    total_exposure = sum(position_values) / portfolio_value
    if total_exposure > 0.95:  # 95% max
        scale_all_positions(0.95 / total_exposure)
    
    # Layer 3: Risk Budget
    total_risk = sum(position_risk_amounts)
    max_risk = 0.03 * portfolio_value  # 3% daily max
    if total_risk > max_risk:
        scale_all_positions(max_risk / total_risk)
```

### Stop Loss Logic
```python
# Initial stop (from strategy)
initial_stop = entry_price - (2 * atr)

# Trailing stop (if profitable)
if current_price > entry_price:
    trailing_stop = current_price - (1.5 * atr)
    stop_loss = max(initial_stop, trailing_stop)

# Hard stop (risk management)
max_loss = entry_price * 0.95  # 5% max loss
stop_loss = max(stop_loss, max_loss)
```

## 5️⃣ Trade Execution Logic

### The Actual Trading Algorithm

```python
def daily_trading_loop(date):
    # Morning: Calculate overnight P&L
    for position in current_positions:
        overnight_pnl = shares * (today_open - yesterday_close)
        equity += overnight_pnl
    
    # 1. Generate new signals
    for symbol in universe:
        data = market_data[symbol]
        signals = strategy.generate_signals(data)
        
    # 2. Determine target positions
    target_allocations = allocate_signals(signals, equity, rules)
    
    # 3. Apply risk limits
    adjusted_allocations = risk_manager.validate(target_allocations)
    
    # 4. Calculate trades needed
    for symbol in all_symbols:
        current_shares = positions.get(symbol, 0)
        target_shares = adjusted_allocations.get(symbol, 0)
        shares_to_trade = target_shares - current_shares
        
        if abs(shares_to_trade) > 0:
            execute_trade(symbol, shares_to_trade)
    
    # 5. End of day accounting
    mark_to_market()
    record_equity()
```

### Trade Execution Details
```python
def execute_trade(symbol, shares):
    price = current_prices[symbol]
    
    if shares > 0:  # BUY
        # Check buying power
        cost = shares * price * 1.0005  # Include 5bps cost
        if cash >= cost:
            positions[symbol] += shares
            cash -= cost
            ledger.record_trade(symbol, shares, price, 'BUY')
        else:
            # Reduce order size to available cash
            affordable_shares = floor(cash / (price * 1.0005))
            execute_trade(symbol, affordable_shares)
    
    else:  # SELL
        shares = abs(shares)
        if positions[symbol] >= shares:
            proceeds = shares * price * 0.9995  # Deduct 5bps
            positions[symbol] -= shares
            cash += proceeds
            ledger.record_trade(symbol, -shares, price, 'SELL')
```

## 6️⃣ ML Strategy Selection Logic

### How ML Chooses Strategies

```python
class MLStrategySelector:
    def select_strategy(self, market_features):
        # Extract features
        features = {
            'volatility': calculate_market_volatility(),
            'trend': calculate_trend_strength(),
            'volume': calculate_volume_profile(),
            'correlation': calculate_correlations()
        }
        
        # Predict best strategy
        if features['volatility'] > 0.02:  # High volatility
            return 'volatility'  # Volatility strategy
        elif features['trend'] > 0.7:  # Strong trend
            return 'trend_breakout'  # Trend following
        elif features['correlation'] < 0.3:  # Low correlation
            return 'mean_reversion'  # Mean reversion
        else:
            return 'demo_ma'  # Default MA strategy
```

### ML Performance Tracking
```python
# After each trade
strategy_performance[strategy_name].append({
    'market_condition': features,
    'return': trade_return,
    'success': trade_return > 0
})

# Periodically retrain
if len(performance_history) > 1000:
    retrain_model(performance_history)
```

## 7️⃣ Critical Decision Points

### Entry Decisions
1. **Signal Present?** - Strategy generates signal > 0
2. **Capital Available?** - Have cash for position
3. **Risk Budget?** - Within daily risk limit
4. **Position Limit?** - Under max positions
5. **Correlation Check?** - Not too correlated to existing

### Exit Decisions
1. **Stop Loss Hit?** - Price below stop level
2. **Take Profit Hit?** - Price above target
3. **Signal Reversal?** - Strategy signal changes
4. **Risk Violation?** - Position becomes too large
5. **Rebalance?** - Portfolio needs adjustment

### Position Size Decisions
1. **Account Size** - Determines risk percentage
2. **Volatility (ATR)** - Determines stop distance
3. **Signal Strength** - Affects ranking
4. **Available Capital** - Hard constraint
5. **Risk Limits** - May reduce size

## 8️⃣ Edge Cases & Special Handling

### Insufficient Data
```python
if len(df) < max(fast_period, slow_period):
    return empty_signals()  # No trading
```

### Missing Prices
```python
if current_price is None:
    skip_symbol()  # Can't trade without price
```

### Extreme Volatility
```python
if atr > price * 0.1:  # ATR > 10% of price
    reduce_position_size(0.5)  # Halve position
```

### Correlation Clusters
```python
if correlation_with_existing > 0.8:
    skip_or_reduce()  # Avoid concentration
```

## 9️⃣ Performance Attribution

### How to Interpret Results

```python
# Win Rate
win_rate = winning_trades / total_trades
# > 50% good for trend following
# > 40% acceptable with good risk/reward

# Profit Factor
profit_factor = gross_profits / gross_losses
# > 1.5 excellent
# > 1.2 good
# < 1.0 losing system

# Sharpe Ratio
sharpe = (returns - risk_free_rate) / volatility
# > 1.0 good
# > 1.5 very good
# > 2.0 excellent

# Maximum Drawdown
max_dd = (peak - trough) / peak
# < 10% excellent
# < 20% acceptable
# > 30% concerning
```

## 🎯 System Logic Validation

### Is Our Logic Sound?

**Strengths:**
1. ✅ **Risk-First Design** - Never risks more than predetermined amount
2. ✅ **Volatility Adaptation** - ATR scales with market conditions
3. ✅ **Multi-Strategy** - Different approaches for different markets
4. ✅ **Position Limits** - Prevents over-concentration
5. ✅ **Dynamic Sizing** - Adapts to account size

**Potential Weaknesses:**
1. ⚠️ **Lagging Indicators** - MAs are backward-looking
2. ⚠️ **No Fundamental Data** - Pure technical approach
3. ⚠️ **Static Parameters** - Fixed periods may not be optimal
4. ⚠️ **No Market Regime Detection** - Treats all markets same
5. ⚠️ **Limited Short Selling** - Most strategies long-only

**Improvements Needed:**
1. 🔧 Add market regime detection
2. 🔧 Implement adaptive parameters
3. 🔧 Include fundamental filters
4. 🔧 Add more sophisticated ML
5. 🔧 Implement options strategies

## 📊 Real Example Walkthrough

### Scenario: $10,000 Account Trading AAPL

```python
# Day 1: Signal Generated
AAPL_price = $150
AAPL_atr = $3
fast_ma = $148
slow_ma = $145
signal = 1  # Buy (fast > slow)

# Position Sizing
risk_pct = 0.01  # 1% for $10K account
risk_dollars = $10,000 * 0.01 = $100
stop_distance = 2 * $3 = $6
shares = $100 / $6 = 16 shares

# Risk Check
position_value = 16 * $150 = $2,400
position_pct = $2,400 / $10,000 = 24%
max_allowed = 25%  # For $10K account
# PASS - within limit

# Execute Trade
cost = 16 * $150 * 1.0005 = $2,401.20
cash = $10,000 - $2,401.20 = $7,598.80
positions = {'AAPL': 16}

# Day 5: Stop Loss Check
AAPL_price = $145
stop_loss = $150 - $6 = $144
# Price $145 > Stop $144 - HOLD

# Day 10: Take Profit
AAPL_price = $160
take_profit = $150 + (3 * $3) = $159
# Price $160 > Target $159 - SELL

# Exit Trade
proceeds = 16 * $160 * 0.9995 = $2,558.72
profit = $2,558.72 - $2,401.20 = $157.52
return = $157.52 / $2,401.20 = 6.56%
```

## 🔍 Conclusion

GPT-Trader's trading logic is fundamentally sound with:
- **Clear entry/exit rules** based on technical indicators
- **Proper risk management** limiting losses
- **Adaptive position sizing** based on volatility
- **Multiple validation layers** preventing excessive risk

The system makes decisions based on:
1. **Technical signals** (primary driver)
2. **Risk constraints** (safety limits)
3. **Portfolio rules** (diversification)
4. **Market conditions** (volatility adjustment)

The logic is transparent, testable, and follows established trading principles.

---

**Document Version**: 1.0  
**Last Updated**: August 16, 2025  
**Purpose**: Deep understanding of trading decision logic