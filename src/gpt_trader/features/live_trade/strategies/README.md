# Trading Strategies

Strategy implementations for live and backtested trading.

## Architecture

Strategies follow a common interface and receive market data from coordinator engines:

```
strategies/
├── perps_baseline/   # Perpetuals momentum strategy
├── spot_baseline/    # Spot trading strategies
└── protocols.py      # Strategy interface definitions
```

## Strategy Protocol

All strategies implement:

```python
class StrategyProtocol(Protocol):
    def evaluate(self, market_state: MarketState) -> Signal | None:
        """Generate trading signal from market state."""
        ...

    def configure(self, config: StrategyConfig) -> None:
        """Apply configuration parameters."""
        ...
```

## Built-in Strategies

| Strategy | Market | Description |
|----------|--------|-------------|
| `perps_baseline` | Perpetuals | Moving average crossover with momentum |
| `spot_baseline` | Spot | Buy-and-hold with rebalancing |

## Creating Custom Strategies

1. Create a new package under `strategies/`
2. Implement `StrategyProtocol`
3. Register in the strategy factory
4. Configure via profile YAML

## Related Packages

- `features/live_trade/risk/` - Pre-trade validation
- `orchestration/strategy_orchestrator/` - Strategy lifecycle
