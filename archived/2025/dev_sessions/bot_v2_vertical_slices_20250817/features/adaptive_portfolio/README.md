# 🎯 Adaptive Portfolio Slice

**Configuration-first portfolio management that adapts behavior based on portfolio size**

From $500 micro accounts to $50,000+ large portfolios - one slice handles all growth stages with appropriate risk and strategy adjustments.

## 📋 Overview

The adaptive portfolio slice provides intelligent portfolio management that automatically adjusts:
- **Risk limits** (1-4% daily, 5-30% quarterly)
- **Position sizing** (2-20 positions based on capital)
- **Strategy selection** (1-4 strategies as portfolio grows)
- **Trading frequency** (PDT-compliant for small accounts)

## 🚀 Quick Start

```python
from adaptive_portfolio import run_adaptive_strategy

# Analyze current portfolio and get tier-appropriate recommendations
result = run_adaptive_strategy(
    current_capital=1500,  # Your current portfolio value
    symbols=["AAPL", "MSFT", "GOOGL"],  # Optional symbols to analyze
    positions=[]  # Current positions (optional)
)

print(f"Current tier: {result.current_tier.value}")
print(f"Recommended actions: {result.recommended_actions}")
for signal in result.signals:
    print(f"Signal: {signal.action} {signal.symbol} - ${signal.target_position_size:,.0f}")
```

## 🏗️ Configuration-First Design

All behavior is controlled by external JSON configuration files:

```bash
config/
├── adaptive_portfolio_config.json         # Default balanced config
├── adaptive_portfolio_conservative.json   # Low-risk template
└── adaptive_portfolio_aggressive.json     # High-risk template
```

### Changing Risk Profile
```bash
# Switch to conservative approach
cp config/adaptive_portfolio_conservative.json config/adaptive_portfolio_config.json

# Or create custom config
jq '.tiers.micro.risk.daily_limit_pct = 0.8' config/adaptive_portfolio_config.json
```

## 📊 Portfolio Tiers

### Tier 1: Micro ($500-$2,500)
- **Positions**: 2-3 max
- **Strategies**: 1 (momentum)
- **Risk**: 1% daily, 8% quarterly
- **Trading**: 3 trades/week (PDT compliant)
- **Focus**: Capital preservation + steady growth

### Tier 2: Small ($2,500-$10,000)
- **Positions**: 3-5 max
- **Strategies**: 2 (momentum + mean reversion)
- **Risk**: 1.5% daily, 12% quarterly
- **Trading**: 6 trades/week (PDT compliant)
- **Focus**: Balanced growth

### Tier 3: Medium ($10,000-$25,000)
- **Positions**: 5-8 max
- **Strategies**: 3 (momentum + mean reversion + trend following)
- **Risk**: 2% daily, 15% quarterly
- **Trading**: 10 trades/week
- **Focus**: Growth with diversification

### Tier 4: Large ($25,000+)
- **Positions**: 8-15 max
- **Strategies**: 4 (all strategies + ML enhanced)
- **Risk**: 2.5% daily, 20% quarterly
- **Trading**: Unlimited (no PDT restrictions)
- **Focus**: Optimized returns

## 🔧 Core Components

### 1. Configuration Manager (`config_manager.py`)
- Hot-reloadable configuration
- Parameter validation
- Multiple environment support

### 2. Tier Manager (`tier_manager.py`)
- Automatic tier detection
- Transition management with hysteresis
- Tier-specific validation

### 3. Risk Manager (`risk_manager.py`)
- Adaptive position sizing
- Tier-appropriate stop losses
- Real-time risk monitoring

### 4. Strategy Selector (`strategy_selector.py`)
- Tier-based strategy allocation
- Signal generation and filtering
- Confidence-based position sizing

## 📈 Usage Examples

### Basic Portfolio Analysis
```python
from adaptive_portfolio import run_adaptive_strategy

# Current portfolio analysis
result = run_adaptive_strategy(current_capital=5000)

print(f"Tier: {result.current_tier.value}")
print(f"Max positions: {result.tier_config.positions.max_positions}")
print(f"Daily risk limit: {result.tier_config.risk.daily_limit_pct}%")
```

### Backtesting with Tier Transitions
```python
from adaptive_portfolio import run_adaptive_backtest

# Test how portfolio grows and adapts over time
metrics = run_adaptive_backtest(
    symbols=["AAPL", "MSFT", "GOOGL", "AMZN"],
    start_date="2023-01-01",
    end_date="2024-01-01",
    initial_capital=1000
)

print(f"Total return: {metrics.total_return_pct:.1f}%")
print(f"Tier transitions: {metrics.tier_transitions}")
print(f"Final tier: {metrics.final_tier.value}")
```

### Custom Configuration
```python
from adaptive_portfolio import load_portfolio_config

# Load custom config
config = load_portfolio_config("config/adaptive_portfolio_conservative.json")

# Use with portfolio manager
result = run_adaptive_strategy(
    current_capital=2000,
    config_path="config/adaptive_portfolio_conservative.json"
)
```

## ⚙️ Configuration Options

### Risk Parameters
```json
{
  "risk": {
    "daily_limit_pct": 1.5,        // Max daily loss %
    "quarterly_limit_pct": 12.0,   // Max quarterly loss %
    "position_stop_loss_pct": 6.0, // Stop loss per position
    "max_sector_exposure_pct": 60.0 // Max sector concentration
  }
}
```

### Position Management
```json
{
  "positions": {
    "min": 3,          // Minimum positions
    "max": 5,          // Maximum positions
    "target": 4        // Target positions
  },
  "min_position_size": 500  // Minimum $ per position
}
```

### Trading Rules
```json
{
  "trading": {
    "max_trades_per_week": 6,    // Frequency limit
    "account_type": "cash",       // "cash" or "margin"
    "settlement_days": 2,         // T+2 for cash accounts
    "pdt_compliant": true         // Pattern Day Trading compliance
  }
}
```

## 🛡️ Safety Features

### Built-in Validation
- Position sizing math validation
- Risk limit consistency checks
- Tier range overlap detection
- PDT rule compliance

### Circuit Breakers
- Daily loss limits halt trading
- Position concentration limits
- Market constraint enforcement
- Invalid signal rejection

### Hysteresis Buffering
Prevents frequent tier transitions:
```json
{
  "rebalancing": {
    "tier_transition_buffer_pct": 5.0  // 5% buffer to prevent oscillation
  }
}
```

## 📋 API Reference

### Main Functions

#### `run_adaptive_strategy(current_capital, symbols=None, positions=None, config_path=None)`
Main entry point for portfolio analysis.

**Returns**: `AdaptiveResult` with tier-appropriate recommendations

#### `run_adaptive_backtest(symbols, start_date, end_date, initial_capital=1000, config_path=None)`
Run backtest with adaptive tier management.

**Returns**: `BacktestMetrics` with performance across tiers

#### `load_portfolio_config(config_path=None)`
Load and validate configuration file.

**Returns**: `PortfolioConfig` object

#### `get_current_tier(capital, config_path=None)`
Determine appropriate tier for capital amount.

**Returns**: Tier name string

### Key Types

#### `AdaptiveResult`
- `current_tier`: Current portfolio tier
- `signals`: List of trading signals
- `recommended_actions`: Human-readable recommendations
- `warnings`: Risk warnings
- `tier_transition_needed`: Whether tier change is needed

#### `TradingSignal`
- `symbol`: Stock symbol
- `action`: "BUY", "SELL", or "HOLD"
- `confidence`: Signal confidence (0-1)
- `target_position_size`: Recommended position size ($)
- `reasoning`: Why this signal was generated

## 🔄 Development Workflow

### Testing Configuration Changes
```bash
# Validate new config
python -c "from adaptive_portfolio import validate_portfolio_config; print(validate_portfolio_config('config/my_config.json'))"

# Test with small backtest
python -c "from adaptive_portfolio import run_adaptive_backtest; print(run_adaptive_backtest(['AAPL'], '2024-01-01', '2024-02-01', 1000, 'config/my_config.json'))"
```

### A/B Testing Different Approaches
```python
configs = {
    "conservative": "config/adaptive_portfolio_conservative.json",
    "default": "config/adaptive_portfolio_config.json",
    "aggressive": "config/adaptive_portfolio_aggressive.json"
}

for name, config_path in configs.items():
    metrics = run_adaptive_backtest(
        ["AAPL", "MSFT"], "2023-01-01", "2024-01-01", 1000, config_path
    )
    print(f"{name}: {metrics.total_return_pct:.1f}% return")
```

## 📚 Integration with Other Slices

While this slice is completely self-contained, it can be enhanced by integrating with:

- **`ml_strategy/`**: For ML-enhanced signal generation
- **`market_regime/`**: For regime-aware tier adjustments  
- **`backtest/`**: For more sophisticated backtesting
- **`risk/`**: For advanced risk analytics

## 🎯 Token Efficiency

**Slice Size**: ~600 tokens total
**Navigation Cost**: 50 tokens (this README)
**Usage Cost**: 400-500 tokens per task

This maintains the core principle of vertical slice architecture - complete self-containment with minimal token usage.

---

**Configuration-first design means you can adapt this system to any risk profile or trading style without touching code.**