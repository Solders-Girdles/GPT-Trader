-- Create Indexes for Optimal Performance

-- Trading Schema Indexes
CREATE INDEX idx_positions_symbol ON trading.positions(symbol) WHERE status = 'open';
CREATE INDEX idx_positions_status ON trading.positions(status);
CREATE INDEX idx_positions_opened_at ON trading.positions(opened_at DESC);
CREATE INDEX idx_positions_strategy ON trading.positions(strategy_id) WHERE status = 'open';

CREATE INDEX idx_orders_symbol ON trading.orders(symbol);
CREATE INDEX idx_orders_status ON trading.orders(status) WHERE status NOT IN ('filled', 'cancelled');
CREATE INDEX idx_orders_submitted_at ON trading.orders(submitted_at DESC);
CREATE INDEX idx_orders_broker_id ON trading.orders(broker_order_id) WHERE broker_order_id IS NOT NULL;

CREATE INDEX idx_trades_symbol ON trading.trades(symbol);
CREATE INDEX idx_trades_executed_at ON trading.trades(executed_at DESC);
CREATE INDEX idx_trades_order_id ON trading.trades(order_id);
CREATE INDEX idx_trades_position_id ON trading.trades(position_id);

-- ML Schema Indexes
CREATE INDEX idx_feature_sets_active ON ml.feature_sets(name) WHERE is_active = TRUE;

-- Feature values indexes (TimescaleDB automatically creates time-based indexes)
CREATE INDEX idx_feature_values_symbol ON ml.feature_values(symbol, timestamp DESC);
CREATE INDEX idx_feature_values_feature_set ON ml.feature_values(feature_set_id, timestamp DESC);

CREATE INDEX idx_models_active ON ml.models(model_type, model_name) WHERE is_active = TRUE;
CREATE INDEX idx_models_production ON ml.models(model_type) WHERE is_production = TRUE;
CREATE INDEX idx_models_training_date ON ml.models(training_date DESC);

-- Predictions indexes (TimescaleDB optimized)
CREATE INDEX idx_predictions_symbol ON ml.predictions(symbol, timestamp DESC);
CREATE INDEX idx_predictions_model ON ml.predictions(model_id, timestamp DESC);
CREATE INDEX idx_predictions_type ON ml.predictions(prediction_type, timestamp DESC);

CREATE INDEX idx_market_regimes_timestamp ON ml.market_regimes(timestamp DESC);
CREATE INDEX idx_market_regimes_regime ON ml.market_regimes(regime, timestamp DESC);
CREATE INDEX idx_market_regimes_model ON ml.market_regimes(model_id);

CREATE INDEX idx_strategy_selections_timestamp ON ml.strategy_selections(timestamp DESC);
CREATE INDEX idx_strategy_selections_strategy ON ml.strategy_selections(selected_strategy, timestamp DESC);
CREATE INDEX idx_strategy_selections_model ON ml.strategy_selections(model_id);

-- Portfolio Schema Indexes
-- Snapshots indexes (TimescaleDB optimized)
CREATE INDEX idx_snapshots_daily_pnl ON portfolio.snapshots(timestamp DESC, daily_pnl);

CREATE INDEX idx_optimization_runs_timestamp ON portfolio.optimization_runs(timestamp DESC);
CREATE INDEX idx_optimization_runs_type ON portfolio.optimization_runs(optimization_type, timestamp DESC);

CREATE INDEX idx_rebalancing_events_timestamp ON portfolio.rebalancing_events(timestamp DESC);
CREATE INDEX idx_rebalancing_events_status ON portfolio.rebalancing_events(status) WHERE status = 'pending';
CREATE INDEX idx_rebalancing_events_trigger ON portfolio.rebalancing_events(trigger_type, timestamp DESC);

-- Monitoring Schema Indexes
-- System metrics indexes (TimescaleDB optimized)
CREATE INDEX idx_system_metrics_name ON monitoring.system_metrics(metric_name, timestamp DESC);
CREATE INDEX idx_system_metrics_component ON monitoring.system_metrics(component, timestamp DESC);

CREATE INDEX idx_performance_metrics_timestamp ON monitoring.performance_metrics(timestamp DESC);
CREATE INDEX idx_performance_metrics_period ON monitoring.performance_metrics(period, timestamp DESC);

CREATE INDEX idx_alerts_timestamp ON monitoring.alerts(timestamp DESC);
CREATE INDEX idx_alerts_severity ON monitoring.alerts(severity, timestamp DESC) WHERE NOT resolved;
CREATE INDEX idx_alerts_type ON monitoring.alerts(alert_type, timestamp DESC);
CREATE INDEX idx_alerts_unresolved ON monitoring.alerts(severity) WHERE resolved = FALSE;

CREATE INDEX idx_audit_log_timestamp ON monitoring.audit_log(timestamp DESC);
CREATE INDEX idx_audit_log_user ON monitoring.audit_log(user_id, timestamp DESC);
CREATE INDEX idx_audit_log_action ON monitoring.audit_log(action, timestamp DESC);
CREATE INDEX idx_audit_log_entity ON monitoring.audit_log(entity_type, entity_id);

-- Create composite indexes for common queries
CREATE INDEX idx_positions_symbol_status ON trading.positions(symbol, status);
CREATE INDEX idx_orders_symbol_status ON trading.orders(symbol, status);
CREATE INDEX idx_predictions_symbol_model ON ml.predictions(symbol, model_id, timestamp DESC);

-- Create indexes for JSONB fields (GIN indexes for containment queries)
CREATE INDEX idx_positions_metadata ON trading.positions USING gin(metadata);
CREATE INDEX idx_orders_metadata ON trading.orders USING gin(metadata);
CREATE INDEX idx_features_data ON ml.feature_values USING gin(features);
CREATE INDEX idx_predictions_value ON ml.predictions USING gin(prediction_value);
CREATE INDEX idx_portfolio_positions ON portfolio.snapshots USING gin(positions);
CREATE INDEX idx_optimization_weights ON portfolio.optimization_runs USING gin(optimal_weights);

-- Create partial indexes for frequently filtered queries
CREATE INDEX idx_active_models ON ml.models(model_id) WHERE is_active = TRUE;
CREATE INDEX idx_production_models ON ml.models(model_id) WHERE is_production = TRUE;
CREATE INDEX idx_pending_orders ON trading.orders(symbol, submitted_at) WHERE status = 'pending';
CREATE INDEX idx_open_positions ON trading.positions(symbol, opened_at) WHERE status = 'open';

-- Add check constraints
ALTER TABLE ml.feature_values ADD CONSTRAINT check_quality_score
    CHECK (quality_score >= 0 AND quality_score <= 1);

ALTER TABLE ml.predictions ADD CONSTRAINT check_confidence
    CHECK (confidence >= 0 AND confidence <= 1);

ALTER TABLE ml.market_regimes ADD CONSTRAINT check_regime_confidence
    CHECK (confidence >= 0 AND confidence <= 1);

ALTER TABLE ml.strategy_selections ADD CONSTRAINT check_selection_confidence
    CHECK (confidence >= 0 AND confidence <= 1);

ALTER TABLE portfolio.rebalancing_events ADD CONSTRAINT check_urgency_score
    CHECK (urgency_score >= 0 AND urgency_score <= 1);

-- Add foreign key constraints with CASCADE options
ALTER TABLE trading.trades
    ADD CONSTRAINT fk_trades_order
    FOREIGN KEY (order_id)
    REFERENCES trading.orders(order_id)
    ON DELETE CASCADE;

ALTER TABLE trading.trades
    ADD CONSTRAINT fk_trades_position
    FOREIGN KEY (position_id)
    REFERENCES trading.positions(position_id)
    ON DELETE SET NULL;

-- Create triggers for updated_at timestamps
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_positions_updated_at BEFORE UPDATE ON trading.positions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_orders_updated_at BEFORE UPDATE ON trading.orders
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_feature_sets_updated_at BEFORE UPDATE ON ml.feature_sets
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_models_updated_at BEFORE UPDATE ON ml.models
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
