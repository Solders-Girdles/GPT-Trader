# Risk Integration Guide (gpt_trader)

The spot-first `coinbase-trader` architecture layers risk controls throughout the
execution stack. This guide highlights the active components and how they work
together. Legacy material describing `src/bot/risk/*` has been archived.

## Configuration Sources

- `config/risk/spot_top10.json` – Default guard thresholds for the top Coinbase
  USD pairs. Override with `RISK_CONFIG_PATH` to supply a custom file.
- Environment flags (`DAILY_LOSS_LIMIT`, `VOL_GUARD_MULTIPLIER`, etc.) merge
  into `BotConfig` and flow into the guard classes.
- `RiskConfigModel` (Pydantic) validates env/JSON payloads via
  `RuntimeSettings.snapshot_env`; validation errors cite the original env var or
  JSON field so misconfigurations surface immediately.

## Execution Path

1. `gpt_trader/cli/__init__.py` builds a `BotConfig` with risk-specific overrides via the `run` command.
2. `gpt_trader/orchestration/bootstrap.py` seeds a `ServiceRegistry` for the active
   profile and passes it into `TradingBot`.
3. `gpt_trader/orchestration/trading_bot/bot.py` constructs the
   `LiveExecutionEngine` (defined in `gpt_trader/orchestration/live_execution.py`),
   which wires:
   - `RiskEngine` (`features/live_trade/risk.py`)
   - Guard classes under `features/live_trade/guards/`
  - Account exposure monitors (`monitoring/domain/perps/liquidation.py`)

## Guard Highlights

| Guard | Module | Purpose |
|-------|--------|---------|
| Daily loss | `guards/daily_loss.py` | Halts trading when cumulative loss exceeds configured limit. |
| Volatility | `guards/volatility.py` | Prevents trading during volatility spikes (configurable z-score window). |
| Correlation | `guards/correlation.py` | Blocks correlated exposures above the configured threshold. |
| Mark staleness | `guards/market_data.py` | Ensures price data is fresh before orders submit. |
| Circuit breaker | `guards/circuit_breaker.py` | Tracks consecutive failures and enforces cooling-off periods. |

Each guard exposes `should_continue()`; failures raise `RiskGuard*Error`
instances defined in `features/live_trade/guard_errors.py`.

## Position Sizing & Exposure

- `features/position_sizing/kelly_allocator.py` implements Kelly scaling with
  per-symbol caps.
- `features/live_trade/portfolio_valuation.py` keeps mark-to-market valuations
  current for guard calculations.
- `monitoring/domain/perps/liquidation.py` enforces leverage and margin
  buffers (spot mode keeps leverage at 1x).

## Telemetry

- Guard results emit structured logs (`risk.guards.<name>.recoverable_failures`
  counters, etc.).
- Metrics written to `metrics.json` surface guard state, including
  `risk_daily_loss` and `market_data_staleness_seconds`.

## Extension Points

- Inject a custom `LiveRiskManager` into the `ServiceRegistry` before calling
  `build_bot()` when you need bespoke guard behaviour.
- Extend `TradingBot._init_risk_manager()` for one-off experiments while we build
  the dedicated guard bootstrapper.
- When derivatives become available, supply a derivative-specific risk config
  file; guard thresholds should account for funding rates and leverage.

## Testing

- See `tests/unit/gpt_trader/live_trade/test_risk_guards.py` (if present) or the
  guard-specific tests under `tests/unit/gpt_trader/live_trade/` for regression
  coverage.
- Add targeted tests whenever guard logic changes to maintain deterministic
  behaviour.
