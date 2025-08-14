-- Trading Schema Tables

-- Trading positions
CREATE TABLE trading.positions (
    id SERIAL PRIMARY KEY,
    position_id UUID DEFAULT uuid_generate_v4() UNIQUE NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    quantity DECIMAL(20,8) NOT NULL,
    entry_price DECIMAL(20,8) NOT NULL,
    current_price DECIMAL(20,8),
    market_value DECIMAL(20,8),
    unrealized_pnl DECIMAL(20,8),
    realized_pnl DECIMAL(20,8),
    opened_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    closed_at TIMESTAMP WITH TIME ZONE,
    status trading.position_status DEFAULT 'open',
    strategy_id VARCHAR(100),
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Trading orders
CREATE TABLE trading.orders (
    id SERIAL PRIMARY KEY,
    order_id UUID DEFAULT uuid_generate_v4() UNIQUE NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    side trading.order_side NOT NULL,
    order_type trading.order_type NOT NULL,
    quantity DECIMAL(20,8) NOT NULL,
    limit_price DECIMAL(20,8),
    stop_price DECIMAL(20,8),
    filled_quantity DECIMAL(20,8) DEFAULT 0,
    average_fill_price DECIMAL(20,8),
    status trading.order_status NOT NULL DEFAULT 'pending',
    broker_order_id VARCHAR(100),
    submitted_at TIMESTAMP WITH TIME ZONE,
    filled_at TIMESTAMP WITH TIME ZONE,
    cancelled_at TIMESTAMP WITH TIME ZONE,
    reason TEXT,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Trades/Executions
CREATE TABLE trading.trades (
    id SERIAL PRIMARY KEY,
    trade_id UUID DEFAULT uuid_generate_v4() UNIQUE NOT NULL,
    order_id UUID REFERENCES trading.orders(order_id),
    position_id UUID REFERENCES trading.positions(position_id),
    symbol VARCHAR(20) NOT NULL,
    side trading.order_side NOT NULL,
    quantity DECIMAL(20,8) NOT NULL,
    price DECIMAL(20,8) NOT NULL,
    commission DECIMAL(20,8) DEFAULT 0,
    slippage DECIMAL(20,8) DEFAULT 0,
    executed_at TIMESTAMP WITH TIME ZONE NOT NULL,
    broker_trade_id VARCHAR(100),
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ML Schema Tables

-- Feature sets
CREATE TABLE ml.feature_sets (
    id SERIAL PRIMARY KEY,
    feature_set_id UUID DEFAULT uuid_generate_v4() UNIQUE NOT NULL,
    name VARCHAR(100) NOT NULL UNIQUE,
    description TEXT,
    feature_count INTEGER,
    feature_names JSONB,
    feature_types JSONB,
    configuration JSONB,
    version VARCHAR(20),
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Feature values (time-series optimized)
CREATE TABLE ml.feature_values (
    symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    feature_set_id UUID REFERENCES ml.feature_sets(feature_set_id),
    features JSONB NOT NULL,
    quality_score DECIMAL(5,4),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    PRIMARY KEY (timestamp, symbol, feature_set_id)
);

-- Convert to TimescaleDB hypertable for time-series optimization
SELECT create_hypertable('ml.feature_values', 'timestamp',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE);

-- ML Models
CREATE TABLE ml.models (
    id SERIAL PRIMARY KEY,
    model_id UUID DEFAULT uuid_generate_v4() UNIQUE NOT NULL,
    model_type VARCHAR(50) NOT NULL,
    model_name VARCHAR(100) NOT NULL,
    version VARCHAR(20),
    model_path TEXT,
    model_hash VARCHAR(64),  -- SHA256 hash for integrity
    training_date TIMESTAMP WITH TIME ZONE,
    training_duration_seconds INTEGER,
    training_config JSONB,
    hyperparameters JSONB,
    performance_metrics JSONB,
    validation_metrics JSONB,
    feature_importance JSONB,
    feature_set_id UUID REFERENCES ml.feature_sets(feature_set_id),
    is_active BOOLEAN DEFAULT FALSE,
    is_production BOOLEAN DEFAULT FALSE,
    created_by VARCHAR(100),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Model predictions (time-series optimized)
CREATE TABLE ml.predictions (
    symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    model_id UUID REFERENCES ml.models(model_id),
    prediction_type VARCHAR(50) NOT NULL,
    prediction_value JSONB NOT NULL,
    confidence DECIMAL(5,4),
    prediction_metadata JSONB,
    features_snapshot JSONB,  -- Features used for this prediction
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    PRIMARY KEY (timestamp, symbol, model_id, prediction_type)
);

-- Convert to TimescaleDB hypertable
SELECT create_hypertable('ml.predictions', 'timestamp',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE);

-- Market regimes
CREATE TABLE ml.market_regimes (
    id SERIAL PRIMARY KEY,
    regime_id UUID DEFAULT uuid_generate_v4() UNIQUE NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    regime VARCHAR(50) NOT NULL,
    confidence DECIMAL(5,4),
    transition_probability JSONB,
    regime_features JSONB,
    model_id UUID REFERENCES ml.models(model_id),
    duration_minutes INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Strategy selections
CREATE TABLE ml.strategy_selections (
    id SERIAL PRIMARY KEY,
    selection_id UUID DEFAULT uuid_generate_v4() UNIQUE NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    selected_strategy VARCHAR(100) NOT NULL,
    confidence DECIMAL(5,4),
    strategy_scores JSONB,  -- All strategy scores
    strategy_weights JSONB,  -- Ensemble weights
    regime VARCHAR(50),
    model_id UUID REFERENCES ml.models(model_id),
    selection_metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Portfolio Schema Tables

-- Portfolio snapshots (time-series optimized)
CREATE TABLE portfolio.snapshots (
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    total_value DECIMAL(20,8) NOT NULL,
    cash_balance DECIMAL(20,8) NOT NULL,
    positions_value DECIMAL(20,8) NOT NULL,
    margin_used DECIMAL(20,8) DEFAULT 0,
    buying_power DECIMAL(20,8),
    daily_pnl DECIMAL(20,8),
    total_pnl DECIMAL(20,8),
    positions_count INTEGER,
    positions JSONB,  -- Detailed position data
    allocations JSONB,  -- Asset allocations
    risk_metrics JSONB,  -- VaR, CVaR, etc.
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    PRIMARY KEY (timestamp)
);

-- Convert to TimescaleDB hypertable
SELECT create_hypertable('portfolio.snapshots', 'timestamp',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE);

-- Optimization runs
CREATE TABLE portfolio.optimization_runs (
    id SERIAL PRIMARY KEY,
    optimization_id UUID DEFAULT uuid_generate_v4() UNIQUE NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    optimization_type VARCHAR(50) NOT NULL,
    objective VARCHAR(50) NOT NULL,
    universe JSONB NOT NULL,  -- List of symbols
    constraints JSONB,
    risk_model JSONB,
    correlation_matrix JSONB,
    covariance_matrix JSONB,
    optimal_weights JSONB NOT NULL,
    expected_return DECIMAL(10,6),
    expected_risk DECIMAL(10,6),
    sharpe_ratio DECIMAL(10,6),
    efficient_frontier JSONB,
    execution_time_ms INTEGER,
    solver_status VARCHAR(50),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Rebalancing events
CREATE TABLE portfolio.rebalancing_events (
    id SERIAL PRIMARY KEY,
    rebalancing_id UUID DEFAULT uuid_generate_v4() UNIQUE NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    trigger_type VARCHAR(50) NOT NULL,
    trigger_reason TEXT,
    urgency_score DECIMAL(5,4),
    current_allocation JSONB NOT NULL,
    target_allocation JSONB NOT NULL,
    allocation_drift JSONB,
    trades_required JSONB,
    estimated_cost DECIMAL(20,8),
    estimated_slippage DECIMAL(20,8),
    actual_cost DECIMAL(20,8),
    actual_slippage DECIMAL(20,8),
    execution_strategy VARCHAR(50),  -- VWAP, TWAP, etc.
    status VARCHAR(20) DEFAULT 'pending',
    executed_at TIMESTAMP WITH TIME ZONE,
    cancelled_at TIMESTAMP WITH TIME ZONE,
    cancellation_reason TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Monitoring Schema Tables

-- System metrics (time-series optimized)
CREATE TABLE monitoring.system_metrics (
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    metric_value DECIMAL(20,8) NOT NULL,
    metric_unit VARCHAR(20),
    component VARCHAR(50),
    tags JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    PRIMARY KEY (timestamp, metric_name, component)
);

-- Convert to TimescaleDB hypertable
SELECT create_hypertable('monitoring.system_metrics', 'timestamp',
    chunk_time_interval => INTERVAL '1 hour',
    if_not_exists => TRUE);

-- Performance metrics
CREATE TABLE monitoring.performance_metrics (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    period VARCHAR(20) NOT NULL,  -- daily, weekly, monthly, yearly
    total_return DECIMAL(10,6),
    annualized_return DECIMAL(10,6),
    volatility DECIMAL(10,6),
    sharpe_ratio DECIMAL(10,6),
    sortino_ratio DECIMAL(10,6),
    calmar_ratio DECIMAL(10,6),
    max_drawdown DECIMAL(10,6),
    max_drawdown_duration_days INTEGER,
    win_rate DECIMAL(10,6),
    profit_factor DECIMAL(10,6),
    avg_win DECIMAL(20,8),
    avg_loss DECIMAL(20,8),
    best_trade DECIMAL(20,8),
    worst_trade DECIMAL(20,8),
    total_trades INTEGER,
    winning_trades INTEGER,
    losing_trades INTEGER,
    gross_profit DECIMAL(20,8),
    gross_loss DECIMAL(20,8),
    commission_paid DECIMAL(20,8),
    slippage_cost DECIMAL(20,8),
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Alerts
CREATE TABLE monitoring.alerts (
    id SERIAL PRIMARY KEY,
    alert_id UUID DEFAULT uuid_generate_v4() UNIQUE NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    alert_type VARCHAR(50) NOT NULL,
    severity VARCHAR(20) NOT NULL CHECK (severity IN ('info', 'warning', 'error', 'critical')),
    component VARCHAR(50),
    message TEXT NOT NULL,
    details JSONB,
    threshold_violated JSONB,
    acknowledged BOOLEAN DEFAULT FALSE,
    acknowledged_by VARCHAR(100),
    acknowledged_at TIMESTAMP WITH TIME ZONE,
    resolved BOOLEAN DEFAULT FALSE,
    resolved_at TIMESTAMP WITH TIME ZONE,
    resolution_notes TEXT,
    auto_resolved BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Audit log
CREATE TABLE monitoring.audit_log (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    user_id VARCHAR(100),
    action VARCHAR(100) NOT NULL,
    entity_type VARCHAR(50),
    entity_id VARCHAR(100),
    old_value JSONB,
    new_value JSONB,
    ip_address INET,
    user_agent TEXT,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
