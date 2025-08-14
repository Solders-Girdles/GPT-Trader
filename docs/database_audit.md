# Database Schema Audit & PostgreSQL Migration Plan

**Date:** 2025-08-14  
**Phase:** 2.5 - Day 1  
**Status:** In Progress

## Current SQLite Database Structure

Based on code analysis, the system uses multiple SQLite databases:

### 1. Main Trading Database (`trading.db`)
**Tables:**
- `trades` - Trade execution records
- `positions` - Current positions
- `orders` - Order history
- `portfolio` - Portfolio snapshots
- `performance_metrics` - Performance tracking

### 2. ML Features Database (`ml_features.db`)
**Tables:**
- `feature_sets` - Feature definitions
- `feature_values` - Calculated feature values
- `feature_importance` - Feature importance scores

### 3. ML Models Database (`ml_models.db`)
**Tables:**
- `models` - Model registry
- `model_versions` - Version history
- `training_runs` - Training metadata
- `model_performance` - Performance metrics

### 4. ML Predictions Database (`ml_predictions.db`)
**Tables:**
- `predictions` - Model predictions
- `regime_states` - Market regime detection
- `strategy_selections` - Strategy recommendations

### 5. Portfolio Optimization Database (`portfolio_optimization.db`)
**Tables:**
- `optimization_runs` - Optimization history
- `efficient_frontiers` - Frontier calculations
- `allocations` - Target allocations
- `constraints` - Optimization constraints

### 6. Rebalancing History Database (`rebalancing_history.db`)
**Tables:**
- `rebalancing_events` - Rebalancing triggers
- `rebalancing_costs` - Transaction costs
- `rebalancing_trades` - Executed trades

## PostgreSQL Unified Schema Design

### Core Schema Structure

```sql
-- Create database
CREATE DATABASE gpt_trader;

-- Use database
\c gpt_trader;

-- Enable extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "timescaledb";  -- For time-series optimization
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";  -- For query analysis

-- Create schemas for logical separation
CREATE SCHEMA trading;
CREATE SCHEMA ml;
CREATE SCHEMA portfolio;
CREATE SCHEMA monitoring;
```

### Trading Schema

```sql
-- Trading positions
CREATE TABLE trading.positions (
    id SERIAL PRIMARY KEY,
    position_id UUID DEFAULT uuid_generate_v4(),
    symbol VARCHAR(20) NOT NULL,
    quantity DECIMAL(20,8) NOT NULL,
    entry_price DECIMAL(20,8) NOT NULL,
    current_price DECIMAL(20,8),
    market_value DECIMAL(20,8),
    unrealized_pnl DECIMAL(20,8),
    realized_pnl DECIMAL(20,8),
    opened_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    closed_at TIMESTAMP WITH TIME ZONE,
    status VARCHAR(20) DEFAULT 'open',
    strategy_id VARCHAR(100),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Trading orders
CREATE TABLE trading.orders (
    id SERIAL PRIMARY KEY,
    order_id UUID DEFAULT uuid_generate_v4(),
    symbol VARCHAR(20) NOT NULL,
    side VARCHAR(10) NOT NULL CHECK (side IN ('buy', 'sell')),
    order_type VARCHAR(20) NOT NULL,
    quantity DECIMAL(20,8) NOT NULL,
    limit_price DECIMAL(20,8),
    stop_price DECIMAL(20,8),
    filled_quantity DECIMAL(20,8) DEFAULT 0,
    average_fill_price DECIMAL(20,8),
    status VARCHAR(20) NOT NULL,
    broker_order_id VARCHAR(100),
    submitted_at TIMESTAMP WITH TIME ZONE NOT NULL,
    filled_at TIMESTAMP WITH TIME ZONE,
    cancelled_at TIMESTAMP WITH TIME ZONE,
    reason TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Trades/Executions
CREATE TABLE trading.trades (
    id SERIAL PRIMARY KEY,
    trade_id UUID DEFAULT uuid_generate_v4(),
    order_id UUID REFERENCES trading.orders(order_id),
    symbol VARCHAR(20) NOT NULL,
    side VARCHAR(10) NOT NULL,
    quantity DECIMAL(20,8) NOT NULL,
    price DECIMAL(20,8) NOT NULL,
    commission DECIMAL(20,8),
    executed_at TIMESTAMP WITH TIME ZONE NOT NULL,
    broker_trade_id VARCHAR(100),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for trading schema
CREATE INDEX idx_positions_symbol ON trading.positions(symbol);
CREATE INDEX idx_positions_status ON trading.positions(status);
CREATE INDEX idx_positions_opened_at ON trading.positions(opened_at);
CREATE INDEX idx_orders_symbol ON trading.orders(symbol);
CREATE INDEX idx_orders_status ON trading.orders(status);
CREATE INDEX idx_orders_submitted_at ON trading.orders(submitted_at);
CREATE INDEX idx_trades_symbol ON trading.trades(symbol);
CREATE INDEX idx_trades_executed_at ON trading.trades(executed_at);
```

### ML Schema

```sql
-- Feature sets
CREATE TABLE ml.feature_sets (
    id SERIAL PRIMARY KEY,
    feature_set_id UUID DEFAULT uuid_generate_v4(),
    name VARCHAR(100) NOT NULL UNIQUE,
    description TEXT,
    feature_count INTEGER,
    feature_names JSONB,
    configuration JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Feature values (partitioned by date)
CREATE TABLE ml.feature_values (
    id BIGSERIAL,
    symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    feature_set_id UUID REFERENCES ml.feature_sets(feature_set_id),
    features JSONB NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    PRIMARY KEY (timestamp, symbol)
) PARTITION BY RANGE (timestamp);

-- Create monthly partitions for feature values
CREATE TABLE ml.feature_values_2025_01 PARTITION OF ml.feature_values
    FOR VALUES FROM ('2025-01-01') TO ('2025-02-01');
-- Continue for other months...

-- ML Models
CREATE TABLE ml.models (
    id SERIAL PRIMARY KEY,
    model_id UUID DEFAULT uuid_generate_v4(),
    model_type VARCHAR(50) NOT NULL,
    model_name VARCHAR(100) NOT NULL,
    version VARCHAR(20),
    model_path TEXT,
    training_date TIMESTAMP WITH TIME ZONE,
    training_config JSONB,
    performance_metrics JSONB,
    feature_importance JSONB,
    is_active BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Model predictions (partitioned)
CREATE TABLE ml.predictions (
    id BIGSERIAL,
    prediction_id UUID DEFAULT uuid_generate_v4(),
    model_id UUID REFERENCES ml.models(model_id),
    symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    prediction_type VARCHAR(50),
    prediction_value JSONB,
    confidence DECIMAL(5,4),
    features_used JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    PRIMARY KEY (timestamp, symbol, model_id)
) PARTITION BY RANGE (timestamp);

-- Market regimes
CREATE TABLE ml.market_regimes (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    regime VARCHAR(50) NOT NULL,
    confidence DECIMAL(5,4),
    transition_probability JSONB,
    features JSONB,
    model_id UUID REFERENCES ml.models(model_id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Strategy selections
CREATE TABLE ml.strategy_selections (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    selected_strategy VARCHAR(100) NOT NULL,
    confidence DECIMAL(5,4),
    strategy_weights JSONB,
    regime VARCHAR(50),
    model_id UUID REFERENCES ml.models(model_id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for ML schema
CREATE INDEX idx_feature_values_symbol_timestamp ON ml.feature_values(symbol, timestamp DESC);
CREATE INDEX idx_predictions_symbol_timestamp ON ml.predictions(symbol, timestamp DESC);
CREATE INDEX idx_predictions_model_id ON ml.predictions(model_id);
CREATE INDEX idx_market_regimes_timestamp ON ml.market_regimes(timestamp DESC);
CREATE INDEX idx_strategy_selections_timestamp ON ml.strategy_selections(timestamp DESC);
```

### Portfolio Schema

```sql
-- Portfolio snapshots
CREATE TABLE portfolio.snapshots (
    id SERIAL PRIMARY KEY,
    snapshot_id UUID DEFAULT uuid_generate_v4(),
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    total_value DECIMAL(20,8) NOT NULL,
    cash_balance DECIMAL(20,8) NOT NULL,
    positions_value DECIMAL(20,8) NOT NULL,
    daily_pnl DECIMAL(20,8),
    total_pnl DECIMAL(20,8),
    positions JSONB,
    metrics JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Optimization runs
CREATE TABLE portfolio.optimization_runs (
    id SERIAL PRIMARY KEY,
    optimization_id UUID DEFAULT uuid_generate_v4(),
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    optimization_type VARCHAR(50),
    objective VARCHAR(50),
    constraints JSONB,
    input_data JSONB,
    results JSONB,
    optimal_weights JSONB,
    expected_return DECIMAL(10,6),
    expected_risk DECIMAL(10,6),
    sharpe_ratio DECIMAL(10,6),
    execution_time_ms INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Rebalancing events
CREATE TABLE portfolio.rebalancing_events (
    id SERIAL PRIMARY KEY,
    rebalancing_id UUID DEFAULT uuid_generate_v4(),
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    trigger_type VARCHAR(50),
    trigger_details JSONB,
    current_allocation JSONB,
    target_allocation JSONB,
    trades_required JSONB,
    estimated_cost DECIMAL(20,8),
    actual_cost DECIMAL(20,8),
    status VARCHAR(20),
    executed_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for portfolio schema
CREATE INDEX idx_snapshots_timestamp ON portfolio.snapshots(timestamp DESC);
CREATE INDEX idx_optimization_runs_timestamp ON portfolio.optimization_runs(timestamp DESC);
CREATE INDEX idx_rebalancing_events_timestamp ON portfolio.rebalancing_events(timestamp DESC);
CREATE INDEX idx_rebalancing_events_status ON portfolio.rebalancing_events(status);
```

### Monitoring Schema

```sql
-- System metrics
CREATE TABLE monitoring.system_metrics (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    metric_value DECIMAL(20,8),
    tags JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Performance metrics
CREATE TABLE monitoring.performance_metrics (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    metric_type VARCHAR(50),
    sharpe_ratio DECIMAL(10,6),
    sortino_ratio DECIMAL(10,6),
    max_drawdown DECIMAL(10,6),
    win_rate DECIMAL(10,6),
    profit_factor DECIMAL(10,6),
    total_return DECIMAL(10,6),
    volatility DECIMAL(10,6),
    details JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Alerts
CREATE TABLE monitoring.alerts (
    id SERIAL PRIMARY KEY,
    alert_id UUID DEFAULT uuid_generate_v4(),
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    alert_type VARCHAR(50),
    severity VARCHAR(20),
    message TEXT,
    details JSONB,
    acknowledged BOOLEAN DEFAULT FALSE,
    acknowledged_by VARCHAR(100),
    acknowledged_at TIMESTAMP WITH TIME ZONE,
    resolved BOOLEAN DEFAULT FALSE,
    resolved_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for monitoring schema
CREATE INDEX idx_system_metrics_timestamp ON monitoring.system_metrics(timestamp DESC);
CREATE INDEX idx_system_metrics_name ON monitoring.system_metrics(metric_name, timestamp DESC);
CREATE INDEX idx_performance_metrics_timestamp ON monitoring.performance_metrics(timestamp DESC);
CREATE INDEX idx_alerts_timestamp ON monitoring.alerts(timestamp DESC);
CREATE INDEX idx_alerts_severity ON monitoring.alerts(severity) WHERE NOT resolved;
```

## Migration Statistics

### Current SQLite Databases:
- **Total Databases:** 6
- **Total Tables:** ~24
- **Estimated Total Records:** Unknown (needs data audit)
- **Total Size:** Unknown (needs size check)

### Target PostgreSQL Structure:
- **Schemas:** 4 (trading, ml, portfolio, monitoring)
- **Tables:** 20
- **Partitioned Tables:** 2 (ml.feature_values, ml.predictions)
- **Indexes:** 25+
- **Extensions:** 3 (uuid-ossp, timescaledb, pg_stat_statements)

## Benefits of PostgreSQL Migration

1. **Concurrency:** Handle 1000+ concurrent connections vs SQLite's single writer
2. **Performance:** Better query optimization, parallel queries
3. **Scalability:** Horizontal scaling with read replicas
4. **Time-series:** TimescaleDB extension for optimized time-series data
5. **JSONB:** Efficient storage and querying of JSON data
6. **Partitioning:** Automatic management of historical data
7. **Full-text Search:** Built-in text search capabilities
8. **Monitoring:** pg_stat_statements for query analysis

## Migration Risk Assessment

### Risks:
1. **Data Loss:** Potential for data corruption during migration
2. **Downtime:** System unavailable during migration
3. **Compatibility:** SQL dialect differences
4. **Performance:** Initial performance tuning required

### Mitigations:
1. **Backups:** Full backup before migration
2. **Validation:** Checksums and row count validation
3. **Rollback Plan:** Keep SQLite as fallback for 2 weeks
4. **Testing:** Thorough testing in staging environment

## Next Steps

1. ✅ Complete database audit
2. ⏳ Setup PostgreSQL development environment
3. ⏳ Write migration scripts
4. ⏳ Test migration with sample data
5. ⏳ Validate data integrity
6. ⏳ Update application code
7. ⏳ Performance testing
8. ⏳ Production migration