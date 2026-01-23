# Feature Flags Reference

This document catalogs all feature flags in GPT-Trader, their canonical sources, and precedence rules.

## Precedence Rules

Configuration values follow this precedence (highest to lowest):

1. **CLI Arguments** - Command-line flags override all other sources
2. **Profile** - Settings from loaded profile (e.g., `--profile dev`)
3. **Environment Variables** - `RISK_*` prefixed vars take precedence over unprefixed
4. **Defaults** - Hardcoded defaults in dataclass definitions

## Canonical Sources

| Flag Type | Canonical Location |
|-----------|-------------------|
| Broker/Mode flags | `BotConfig` |
| Strategy flags | `PerpsStrategyConfig` or `MeanReversionConfig` (based on `strategy_type`) |
| Risk runtime config | Derived into `RiskConfig` at runtime |

## Feature Flag Matrix

### Broker & Mode Flags

| Flag | Canonical Source | Env Var | CLI | Default | Notes |
|------|------------------|---------|-----|---------|-------|
| `mock_broker` | `BotConfig` | `MOCK_BROKER` | `--mock-broker` | `False` | Use deterministic broker for testing |
| `dry_run` | `BotConfig` | `DRY_RUN` | `--dry-run` | `False` | Run without placing real orders |
| `status_enabled` | `BotConfig` | `STATUS_ENABLED` | - | `True` | Enable status file updates |
| `perps_paper_trading` | `BotConfig` | `PERPS_PAPER` | - | `False` | Paper trading mode |

### Derivatives & Products

| Flag | Canonical Source | Env Var | CLI | Default | Notes |
|------|------------------|---------|-----|---------|-------|
| `derivatives_enabled` | `BotConfig` | `COINBASE_ENABLE_INTX_PERPS` (legacy: `COINBASE_ENABLE_DERIVATIVES`) | - | `False` | Enable INTX perpetuals (international) |
| `cfm_enabled` | `BotConfig` | `CFM_ENABLED` | - | `False` | Enable CFM futures |
| `trading_modes` | `BotConfig` | `TRADING_MODES` | - | `["spot"]` | Markets to trade: spot, cfm |

### Execution Control

| Flag | Canonical Source | Env Var | CLI | Default | Notes |
|------|------------------|---------|-----|---------|-------|
| `enable_order_preview` | `BotConfig` | `ORDER_PREVIEW_ENABLED` | `--enable-preview` | `False` | Pre-trade validation |
| `reduce_only_mode` | `BotConfig` | `RISK_REDUCE_ONLY_MODE` | `--reduce-only` | `False` | Only allow reducing positions |
| `order_submission_retries_enabled` | `BotConfig` | `ORDER_SUBMISSION_RETRIES_ENABLED` | - | `False` | Retry order submissions on transient failures (off by default) |
| `broker_calls_use_dedicated_executor` | `BotConfig` | `BROKER_CALLS_USE_DEDICATED_EXECUTOR` | - | `False` | Use a dedicated thread pool for broker calls |

### Strategy Flags

These flags are **canonicalized on strategy config** based on `strategy_type`:
- For `baseline`/`ensemble`: Use `config.strategy.*`
- For `mean_reversion`: Use `config.mean_reversion.*`

| Flag | Canonical Source | Env Var | Default | Notes |
|------|------------------|---------|---------|-------|
| `kill_switch_enabled` | Strategy config | - | `False` | Emergency stop all trading |
| `enable_shorts` | Strategy config | - | Varies | Allow short positions |
| `force_entry_on_trend` | Strategy config | - | `False` | Allow trend entries |

**Sync Warning**: If `BotConfig.enable_shorts` differs from the strategy config value, a warning is emitted. The strategy config value is canonical.

### API Resilience Flags

| Flag | Canonical Source | Env Var | Default | Notes |
|------|------------------|---------|---------|-------|
| Cache | `constants.py` | `GPT_TRADER_CACHE_ENABLED` | `True` | Response caching |
| Circuit Breaker | `constants.py` | `GPT_TRADER_CIRCUIT_BREAKER_ENABLED` | `True` | Prevent hammering failed endpoints |
| Priority Queue | `constants.py` | `GPT_TRADER_PRIORITY_ENABLED` | `True` | Prioritize critical requests |
| Adaptive Throttle | `constants.py` | `GPT_TRADER_ADAPTIVE_THROTTLE` | `True` | Proactive rate limiting |
| Metrics | `constants.py` | `GPT_TRADER_METRICS_ENABLED` | `True` | In-memory metrics collection |

### Observability Flags

| Flag | Canonical Source | Env Var | Default | Notes |
|------|------------------|---------|---------|-------|
| Metrics Endpoint | `constants.py` | `GPT_TRADER_METRICS_ENDPOINT_ENABLED` | `False` | Expose /metrics endpoint |
| OTEL Tracing | `constants.py` | `GPT_TRADER_OTEL_ENABLED` | `False` | OpenTelemetry tracing |

### Risk Guard Thresholds

| Setting | Env Var | Default | Notes |
|---------|---------|---------|-------|
| Daily Loss Limit | `RISK_DAILY_LOSS_LIMIT_PCT` | `0.05` | 5% of equity |
| API Error Rate | `RISK_API_ERROR_RATE_THRESHOLD` | `0.2` | 20% triggers guard |
| Rate Limit Usage | `RISK_API_RATE_LIMIT_USAGE_THRESHOLD` | `0.9` | 90% triggers guard |
| Vol Reduce-Only | `RISK_VOL_REDUCE_ONLY_THRESH` | `0.22` | 22% triggers reduce-only |
| Vol Kill Switch | `RISK_VOL_KILL_SWITCH_THRESH` | `0.26` | 26% triggers kill switch |

### Health Thresholds

| Setting | Env Var | Default | Notes |
|---------|---------|---------|-------|
| Order Error Rate Warn | `HEALTH_ORDER_ERROR_RATE_WARN` | `0.05` | Warning threshold for order errors |
| Order Error Rate Crit | `HEALTH_ORDER_ERROR_RATE_CRIT` | `0.15` | Critical threshold for order errors |
| Order Retry Rate Warn | `HEALTH_ORDER_RETRY_RATE_WARN` | `0.10` | Warning threshold for retry rate |
| Order Retry Rate Crit | `HEALTH_ORDER_RETRY_RATE_CRIT` | `0.25` | Critical threshold for retry rate |
| Broker Latency Warn (ms) | `HEALTH_BROKER_LATENCY_MS_WARN` | `1000.0` | Warning threshold for broker latency |
| Broker Latency Crit (ms) | `HEALTH_BROKER_LATENCY_MS_CRIT` | `3000.0` | Critical threshold for broker latency |
| WS Staleness Warn (s) | `HEALTH_WS_STALENESS_SECONDS_WARN` | `30.0` | Warning threshold for WebSocket staleness |
| WS Staleness Crit (s) | `HEALTH_WS_STALENESS_SECONDS_CRIT` | `60.0` | Critical threshold for WebSocket staleness |
| Guard Trip Count Warn | `HEALTH_GUARD_TRIP_COUNT_WARN` | `3` | Warning threshold for guard trip count |
| Guard Trip Count Crit | `HEALTH_GUARD_TRIP_COUNT_CRIT` | `10` | Critical threshold for guard trip count |

## Deprecated Flags

None.

## Runtime Derivation

Some flags are derived at runtime when building `RiskConfig` for `LiveRiskManager`:

```python
# In RiskValidationContainer.risk_manager:
if config.strategy_type == "mean_reversion":
    kill_switch = config.mean_reversion.kill_switch_enabled
else:
    kill_switch = config.strategy.kill_switch_enabled

risk_config = RiskConfig(
    kill_switch_enabled=kill_switch,
    reduce_only_mode=config.reduce_only_mode,
    ...
)
```

## Testing

Tests for feature flag precedence and sync warnings are in:
- `tests/unit/gpt_trader/app/config/test_bot_config.py`
- `tests/unit/gpt_trader/app/containers/test_risk_validation.py`

Run with:
```bash
uv run pytest tests/unit/gpt_trader/app/config/ tests/unit/gpt_trader/app/containers/ -v
```
