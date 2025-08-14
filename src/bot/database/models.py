"""
PostgreSQL Database Models using SQLAlchemy
Phase 2.5 - Production Database Models

Unified schema models for GPT-Trader system replacing multiple SQLite databases.
"""

import enum
from datetime import datetime
from uuid import uuid4

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
)
from sqlalchemy import (
    Enum as SQLEnum,
)
from sqlalchemy.dialects.postgresql import INET, JSONB, UUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, validates

Base = declarative_base()


# Enums
class OrderSide(enum.Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(enum.Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderStatus(enum.Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class PositionStatus(enum.Enum):
    OPEN = "open"
    CLOSED = "closed"
    PARTIAL = "partial"


class AlertSeverity(enum.Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


# Trading Schema Models


class Position(Base):
    __tablename__ = "positions"
    __table_args__ = {"schema": "trading"}

    id = Column(Integer, primary_key=True)
    position_id = Column(UUID(as_uuid=True), default=uuid4, unique=True, nullable=False)
    symbol = Column(String(20), nullable=False, index=True)
    quantity = Column(Numeric(20, 8), nullable=False)
    entry_price = Column(Numeric(20, 8), nullable=False)
    current_price = Column(Numeric(20, 8))
    market_value = Column(Numeric(20, 8))
    unrealized_pnl = Column(Numeric(20, 8))
    realized_pnl = Column(Numeric(20, 8))
    opened_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
    closed_at = Column(DateTime(timezone=True))
    status = Column(SQLEnum(PositionStatus), default=PositionStatus.OPEN, index=True)
    strategy_id = Column(String(100), index=True)
    metadata = Column(JSONB)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    trades = relationship("Trade", back_populates="position")

    def __repr__(self):
        return f"<Position({self.symbol}, {self.quantity}, {self.status.value})>"

    @validates("quantity")
    def validate_quantity(self, key, value):
        if value <= 0:
            raise ValueError("Quantity must be positive")
        return value


class Order(Base):
    __tablename__ = "orders"
    __table_args__ = {"schema": "trading"}

    id = Column(Integer, primary_key=True)
    order_id = Column(UUID(as_uuid=True), default=uuid4, unique=True, nullable=False)
    symbol = Column(String(20), nullable=False, index=True)
    side = Column(SQLEnum(OrderSide), nullable=False)
    order_type = Column(SQLEnum(OrderType), nullable=False)
    quantity = Column(Numeric(20, 8), nullable=False)
    limit_price = Column(Numeric(20, 8))
    stop_price = Column(Numeric(20, 8))
    filled_quantity = Column(Numeric(20, 8), default=0)
    average_fill_price = Column(Numeric(20, 8))
    status = Column(SQLEnum(OrderStatus), nullable=False, default=OrderStatus.PENDING, index=True)
    broker_order_id = Column(String(100), index=True)
    submitted_at = Column(DateTime(timezone=True))
    filled_at = Column(DateTime(timezone=True))
    cancelled_at = Column(DateTime(timezone=True))
    reason = Column(Text)
    metadata = Column(JSONB)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    trades = relationship("Trade", back_populates="order")

    def __repr__(self):
        return f"<Order({self.symbol}, {self.side.value}, {self.quantity}, {self.status.value})>"


class Trade(Base):
    __tablename__ = "trades"
    __table_args__ = {"schema": "trading"}

    id = Column(Integer, primary_key=True)
    trade_id = Column(UUID(as_uuid=True), default=uuid4, unique=True, nullable=False)
    order_id = Column(UUID(as_uuid=True), ForeignKey("trading.orders.order_id"))
    position_id = Column(UUID(as_uuid=True), ForeignKey("trading.positions.position_id"))
    symbol = Column(String(20), nullable=False, index=True)
    side = Column(SQLEnum(OrderSide), nullable=False)
    quantity = Column(Numeric(20, 8), nullable=False)
    price = Column(Numeric(20, 8), nullable=False)
    commission = Column(Numeric(20, 8), default=0)
    slippage = Column(Numeric(20, 8), default=0)
    executed_at = Column(DateTime(timezone=True), nullable=False, index=True)
    broker_trade_id = Column(String(100))
    metadata = Column(JSONB)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)

    # Relationships
    order = relationship("Order", back_populates="trades")
    position = relationship("Position", back_populates="trades")

    def __repr__(self):
        return f"<Trade({self.symbol}, {self.side.value}, {self.quantity}@{self.price})>"


# ML Schema Models


class FeatureSet(Base):
    __tablename__ = "feature_sets"
    __table_args__ = {"schema": "ml"}

    id = Column(Integer, primary_key=True)
    feature_set_id = Column(UUID(as_uuid=True), default=uuid4, unique=True, nullable=False)
    name = Column(String(100), nullable=False, unique=True)
    description = Column(Text)
    feature_count = Column(Integer)
    feature_names = Column(JSONB)
    feature_types = Column(JSONB)
    configuration = Column(JSONB)
    version = Column(String(20))
    is_active = Column(Boolean, default=True, index=True)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    feature_values = relationship("FeatureValue", back_populates="feature_set")
    models = relationship("Model", back_populates="feature_set")

    def __repr__(self):
        return f"<FeatureSet({self.name}, v{self.version}, {self.feature_count} features)>"


class FeatureValue(Base):
    __tablename__ = "feature_values"
    __table_args__ = (
        Index("idx_feature_values_symbol_timestamp", "symbol", "timestamp"),
        {"schema": "ml"},
    )

    symbol = Column(String(20), primary_key=True)
    timestamp = Column(DateTime(timezone=True), primary_key=True)
    feature_set_id = Column(
        UUID(as_uuid=True), ForeignKey("ml.feature_sets.feature_set_id"), primary_key=True
    )
    features = Column(JSONB, nullable=False)
    quality_score = Column(Numeric(5, 4))
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)

    # Relationships
    feature_set = relationship("FeatureSet", back_populates="feature_values")

    __table_args__ = (
        CheckConstraint("quality_score >= 0 AND quality_score <= 1", name="check_quality_score"),
        Index("idx_feature_values_symbol_timestamp", "symbol", "timestamp"),
        {"schema": "ml"},
    )

    def __repr__(self):
        return f"<FeatureValue({self.symbol}, {self.timestamp})>"


class Model(Base):
    __tablename__ = "models"
    __table_args__ = {"schema": "ml"}

    id = Column(Integer, primary_key=True)
    model_id = Column(UUID(as_uuid=True), default=uuid4, unique=True, nullable=False)
    model_type = Column(String(50), nullable=False, index=True)
    model_name = Column(String(100), nullable=False)
    version = Column(String(20))
    model_path = Column(Text)
    model_hash = Column(String(64))  # SHA256 hash
    training_date = Column(DateTime(timezone=True))
    training_duration_seconds = Column(Integer)
    training_config = Column(JSONB)
    hyperparameters = Column(JSONB)
    performance_metrics = Column(JSONB)
    validation_metrics = Column(JSONB)
    feature_importance = Column(JSONB)
    feature_set_id = Column(UUID(as_uuid=True), ForeignKey("ml.feature_sets.feature_set_id"))
    is_active = Column(Boolean, default=False, index=True)
    is_production = Column(Boolean, default=False, index=True)
    created_by = Column(String(100))
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    feature_set = relationship("FeatureSet", back_populates="models")
    predictions = relationship("Prediction", back_populates="model")
    market_regimes = relationship("MarketRegime", back_populates="model")
    strategy_selections = relationship("StrategySelection", back_populates="model")

    def __repr__(self):
        return f"<Model({self.model_type}, {self.model_name}, v{self.version})>"


class Prediction(Base):
    __tablename__ = "predictions"
    __table_args__ = (
        Index("idx_predictions_symbol_timestamp", "symbol", "timestamp"),
        {"schema": "ml"},
    )

    symbol = Column(String(20), primary_key=True)
    timestamp = Column(DateTime(timezone=True), primary_key=True)
    model_id = Column(UUID(as_uuid=True), ForeignKey("ml.models.model_id"), primary_key=True)
    prediction_type = Column(String(50), primary_key=True)
    prediction_value = Column(JSONB, nullable=False)
    confidence = Column(Numeric(5, 4))
    prediction_metadata = Column(JSONB)
    features_snapshot = Column(JSONB)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)

    # Relationships
    model = relationship("Model", back_populates="predictions")

    __table_args__ = (
        CheckConstraint("confidence >= 0 AND confidence <= 1", name="check_prediction_confidence"),
        Index("idx_predictions_symbol_timestamp", "symbol", "timestamp"),
        Index("idx_predictions_model", "model_id", "timestamp"),
        {"schema": "ml"},
    )

    def __repr__(self):
        return f"<Prediction({self.symbol}, {self.prediction_type}, conf={self.confidence})>"


class MarketRegime(Base):
    __tablename__ = "market_regimes"
    __table_args__ = {"schema": "ml"}

    id = Column(Integer, primary_key=True)
    regime_id = Column(UUID(as_uuid=True), default=uuid4, unique=True, nullable=False)
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    regime = Column(String(50), nullable=False, index=True)
    confidence = Column(Numeric(5, 4))
    transition_probability = Column(JSONB)
    regime_features = Column(JSONB)
    model_id = Column(UUID(as_uuid=True), ForeignKey("ml.models.model_id"))
    duration_minutes = Column(Integer)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)

    # Relationships
    model = relationship("Model", back_populates="market_regimes")

    __table_args__ = (
        CheckConstraint("confidence >= 0 AND confidence <= 1", name="check_regime_confidence"),
        {"schema": "ml"},
    )

    def __repr__(self):
        return f"<MarketRegime({self.regime}, conf={self.confidence})>"


class StrategySelection(Base):
    __tablename__ = "strategy_selections"
    __table_args__ = {"schema": "ml"}

    id = Column(Integer, primary_key=True)
    selection_id = Column(UUID(as_uuid=True), default=uuid4, unique=True, nullable=False)
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    selected_strategy = Column(String(100), nullable=False, index=True)
    confidence = Column(Numeric(5, 4))
    strategy_scores = Column(JSONB)
    strategy_weights = Column(JSONB)
    regime = Column(String(50))
    model_id = Column(UUID(as_uuid=True), ForeignKey("ml.models.model_id"))
    selection_metadata = Column(JSONB)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)

    # Relationships
    model = relationship("Model", back_populates="strategy_selections")

    __table_args__ = (
        CheckConstraint("confidence >= 0 AND confidence <= 1", name="check_selection_confidence"),
        {"schema": "ml"},
    )

    def __repr__(self):
        return f"<StrategySelection({self.selected_strategy}, conf={self.confidence})>"


# Portfolio Schema Models


class PortfolioSnapshot(Base):
    __tablename__ = "snapshots"
    __table_args__ = {"schema": "portfolio"}

    timestamp = Column(DateTime(timezone=True), primary_key=True)
    total_value = Column(Numeric(20, 8), nullable=False)
    cash_balance = Column(Numeric(20, 8), nullable=False)
    positions_value = Column(Numeric(20, 8), nullable=False)
    margin_used = Column(Numeric(20, 8), default=0)
    buying_power = Column(Numeric(20, 8))
    daily_pnl = Column(Numeric(20, 8))
    total_pnl = Column(Numeric(20, 8))
    positions_count = Column(Integer)
    positions = Column(JSONB)
    allocations = Column(JSONB)
    risk_metrics = Column(JSONB)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)

    def __repr__(self):
        return f"<PortfolioSnapshot({self.timestamp}, value={self.total_value})>"


class OptimizationRun(Base):
    __tablename__ = "optimization_runs"
    __table_args__ = {"schema": "portfolio"}

    id = Column(Integer, primary_key=True)
    optimization_id = Column(UUID(as_uuid=True), default=uuid4, unique=True, nullable=False)
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    optimization_type = Column(String(50), nullable=False, index=True)
    objective = Column(String(50), nullable=False)
    universe = Column(JSONB, nullable=False)
    constraints = Column(JSONB)
    risk_model = Column(JSONB)
    correlation_matrix = Column(JSONB)
    covariance_matrix = Column(JSONB)
    optimal_weights = Column(JSONB, nullable=False)
    expected_return = Column(Numeric(10, 6))
    expected_risk = Column(Numeric(10, 6))
    sharpe_ratio = Column(Numeric(10, 6))
    efficient_frontier = Column(JSONB)
    execution_time_ms = Column(Integer)
    solver_status = Column(String(50))
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)

    def __repr__(self):
        return f"<OptimizationRun({self.optimization_type}, sharpe={self.sharpe_ratio})>"


class RebalancingEvent(Base):
    __tablename__ = "rebalancing_events"
    __table_args__ = {"schema": "portfolio"}

    id = Column(Integer, primary_key=True)
    rebalancing_id = Column(UUID(as_uuid=True), default=uuid4, unique=True, nullable=False)
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    trigger_type = Column(String(50), nullable=False, index=True)
    trigger_reason = Column(Text)
    urgency_score = Column(Numeric(5, 4))
    current_allocation = Column(JSONB, nullable=False)
    target_allocation = Column(JSONB, nullable=False)
    allocation_drift = Column(JSONB)
    trades_required = Column(JSONB)
    estimated_cost = Column(Numeric(20, 8))
    estimated_slippage = Column(Numeric(20, 8))
    actual_cost = Column(Numeric(20, 8))
    actual_slippage = Column(Numeric(20, 8))
    execution_strategy = Column(String(50))
    status = Column(String(20), default="pending", index=True)
    executed_at = Column(DateTime(timezone=True))
    cancelled_at = Column(DateTime(timezone=True))
    cancellation_reason = Column(Text)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)

    __table_args__ = (
        CheckConstraint("urgency_score >= 0 AND urgency_score <= 1", name="check_urgency_score"),
        {"schema": "portfolio"},
    )

    def __repr__(self):
        return f"<RebalancingEvent({self.trigger_type}, status={self.status})>"


# Monitoring Schema Models


class SystemMetric(Base):
    __tablename__ = "system_metrics"
    __table_args__ = (
        Index("idx_system_metrics_name_timestamp", "metric_name", "timestamp"),
        {"schema": "monitoring"},
    )

    timestamp = Column(DateTime(timezone=True), primary_key=True)
    metric_name = Column(String(100), primary_key=True)
    component = Column(String(50), primary_key=True)
    metric_value = Column(Numeric(20, 8), nullable=False)
    metric_unit = Column(String(20))
    tags = Column(JSONB)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)

    def __repr__(self):
        return f"<SystemMetric({self.metric_name}, {self.metric_value}{self.metric_unit})>"


class PerformanceMetric(Base):
    __tablename__ = "performance_metrics"
    __table_args__ = {"schema": "monitoring"}

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    period = Column(String(20), nullable=False, index=True)
    total_return = Column(Numeric(10, 6))
    annualized_return = Column(Numeric(10, 6))
    volatility = Column(Numeric(10, 6))
    sharpe_ratio = Column(Numeric(10, 6))
    sortino_ratio = Column(Numeric(10, 6))
    calmar_ratio = Column(Numeric(10, 6))
    max_drawdown = Column(Numeric(10, 6))
    max_drawdown_duration_days = Column(Integer)
    win_rate = Column(Numeric(10, 6))
    profit_factor = Column(Numeric(10, 6))
    avg_win = Column(Numeric(20, 8))
    avg_loss = Column(Numeric(20, 8))
    best_trade = Column(Numeric(20, 8))
    worst_trade = Column(Numeric(20, 8))
    total_trades = Column(Integer)
    winning_trades = Column(Integer)
    losing_trades = Column(Integer)
    gross_profit = Column(Numeric(20, 8))
    gross_loss = Column(Numeric(20, 8))
    commission_paid = Column(Numeric(20, 8))
    slippage_cost = Column(Numeric(20, 8))
    metadata = Column(JSONB)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)

    def __repr__(self):
        return f"<PerformanceMetric({self.period}, sharpe={self.sharpe_ratio})>"


class Alert(Base):
    __tablename__ = "alerts"
    __table_args__ = {"schema": "monitoring"}

    id = Column(Integer, primary_key=True)
    alert_id = Column(UUID(as_uuid=True), default=uuid4, unique=True, nullable=False)
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    alert_type = Column(String(50), nullable=False, index=True)
    severity = Column(SQLEnum(AlertSeverity), nullable=False, index=True)
    component = Column(String(50))
    message = Column(Text, nullable=False)
    details = Column(JSONB)
    threshold_violated = Column(JSONB)
    acknowledged = Column(Boolean, default=False)
    acknowledged_by = Column(String(100))
    acknowledged_at = Column(DateTime(timezone=True))
    resolved = Column(Boolean, default=False, index=True)
    resolved_at = Column(DateTime(timezone=True))
    resolution_notes = Column(Text)
    auto_resolved = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)

    def __repr__(self):
        return f"<Alert({self.alert_type}, {self.severity.value}, resolved={self.resolved})>"


class AuditLog(Base):
    __tablename__ = "audit_log"
    __table_args__ = {"schema": "monitoring"}

    id = Column(BigInteger, primary_key=True)
    timestamp = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow, index=True)
    user_id = Column(String(100), index=True)
    action = Column(String(100), nullable=False, index=True)
    entity_type = Column(String(50), index=True)
    entity_id = Column(String(100), index=True)
    old_value = Column(JSONB)
    new_value = Column(JSONB)
    ip_address = Column(INET)
    user_agent = Column(Text)
    metadata = Column(JSONB)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)

    def __repr__(self):
        return f"<AuditLog({self.action}, {self.entity_type}:{self.entity_id})>"
