"""
Centralized test constants for use across all test modules.

This module provides shared constants, test values, and configuration
to ensure consistency and reduce duplication across the test suite.
"""

from datetime import datetime, timedelta
from decimal import Decimal

# Trading symbols
TEST_SYMBOLS = {
    "BTC": "BTC-USD",
    "ETH": "ETH-USD",
    "AAPL": "AAPL",
    "GOOGL": "GOOGL",
    "MSFT": "MSFT",
    "AMZN": "AMZN",
    "TSLA": "TSLA",
}

# Default test symbols
DEFAULT_TEST_SYMBOL = TEST_SYMBOLS["BTC"]
DEFAULT_SPOT_SYMBOL = TEST_SYMBOLS["AAPL"]

# Test prices and quantities
DEFAULT_TEST_PRICE = Decimal("50000.0")
DEFAULT_TEST_QUANTITY = Decimal("1.0")
DEFAULT_SPOT_PRICE = Decimal("150.0")
DEFAULT_SPOT_QUANTITY = Decimal("100")

# Market data constants
DEFAULT_START_DATE = "2023-01-01"
DEFAULT_END_DATE = "2023-12-31"
DEFAULT_INITIAL_PRICE = 100.0
DEFAULT_DRIFT = 0.0005
DEFAULT_VOLATILITY = 0.02

# Risk management constants
DEFAULT_LEVERAGE = 2
DEFAULT_MAX_POSITION_SIZE = Decimal("1.0")
DEFAULT_MIN_LIQUIDATION_BUFFER = Decimal("0.15")
DEFAULT_DAILY_LOSS_LIMIT = Decimal("500.0")

# Portfolio constants
DEFAULT_INITIAL_CASH = Decimal("100000.0")
DEFAULT_RISK_PER_TRADE = Decimal("0.02")
DEFAULT_MAX_POSITIONS = 5

# Test timestamps
TEST_START_TIME = datetime(2023, 1, 1, 9, 30, 0)
TEST_END_TIME = datetime(2023, 1, 1, 16, 0, 0)
TEST_TIME_DELTA = timedelta(seconds=1)

# Order constants
DEFAULT_ORDER_ID = "test_order_001"
DEFAULT_CLIENT_ID = "test_client_001"
DEFAULT_ORDER_TIMEOUT = 30  # seconds

# Test market scenarios
MARKET_SCENARIOS = {
    "normal": {"volatility": 0.02, "liquidity": "high", "trend": "sideways"},
    "bull": {"volatility": 0.015, "liquidity": "high", "trend": "up"},
    "bear": {"volatility": 0.025, "liquidity": "medium", "trend": "down"},
    "volatile": {"volatility": 0.04, "liquidity": "medium", "trend": "sideways"},
    "crash": {"volatility": 0.05, "liquidity": "low", "trend": "down"},
    "recovery": {"volatility": 0.02, "liquidity": "high", "trend": "up"},
    "low_liquidity": {"volatility": 0.03, "liquidity": "low", "trend": "down"},
    "flash_crash": {"volatility": 0.50, "liquidity": "very_low", "trend": "crash"},
}

# Circuit breaker test scenarios
CIRCUIT_BREAKER_SCENARIOS = {
    "daily_loss_breach": {
        "current_loss": Decimal("600.0"),
        "daily_limit": Decimal("500.0"),
        "expected_action": "stop_trading",
    },
    "liquidation_buffer_breach": {
        "buffer_ratio": Decimal("0.08"),
        "min_buffer": Decimal("0.15"),
        "expected_action": "reduce_positions",
    },
    "volatility_spike": {
        "current_volatility": Decimal("0.25"),
        "volatility_threshold": Decimal("0.10"),
        "expected_action": "reduce_size",
    },
    "correlation_risk": {
        "correlation": Decimal("0.95"),
        "correlation_limit": Decimal("0.8"),
        "expected_action": "halt_new_positions",
    },
}

# Broker error scenarios
BROKER_ERROR_SCENARIOS = {
    "connection_drop": {
        "error_type": "ConnectionError",
        "recovery_action": "reconnect",
        "expected_orders": "retry",
    },
    "rate_limit": {
        "error_type": "RateLimitError",
        "recovery_action": "backoff",
        "expected_orders": "queue",
    },
    "maintenance": {
        "error_type": "MaintenanceError",
        "recovery_action": "wait",
        "expected_orders": "reject",
    },
    "insufficient_liquidity": {
        "error_type": "LiquidityError",
        "recovery_action": "reduce_size",
        "expected_orders": "modify",
    },
}

# Test configuration values
TEST_TIMEOUT = 30  # seconds
TEST_RETRY_COUNT = 3
TEST_RETRY_DELAY = 1  # second

# Environment variables for testing
TEST_ENV_VARS = {
    "GPT_TRADER_RUNTIME_ROOT": "/tmp/test_runtime",
    "EVENT_STORE_ROOT": "/tmp/test_events",
    "COINBASE_DEFAULT_QUOTE": "usd",
    "COINBASE_ENABLE_DERIVATIVES": "true",
    "PERPS_ENABLE_STREAMING": "true",
    "PERPS_STREAM_LEVEL": "2",
    "PERPS_PAPER": "false",
    "PERPS_FORCE_MOCK": "true",
    "PERPS_SKIP_RECONCILE": "true",
    "PERPS_POSITION_FRACTION": "0.25",
    "ORDER_PREVIEW_ENABLED": "true",
    "SPOT_FORCE_LIVE": "false",
    "BROKER": "mock",
    "COINBASE_SANDBOX": "true",
    "COINBASE_API_MODE": "sandbox",
    "LOG_LEVEL": "DEBUG",
    "INTEGRATION_TEST_MODE": "true",
}

# Test file paths
TEST_CONFIG_PATH = "/tmp/test_config"
TEST_DATA_PATH = "/tmp/test_data"
TEST_LOG_PATH = "/tmp/test_logs"

# Performance test thresholds
PERFORMANCE_THRESHOLDS = {
    "max_order_latency_ms": 100,
    "max_risk_check_latency_ms": 50,
    "max_position_update_latency_ms": 75,
    "min_throughput_orders_per_sec": 10,
    "max_memory_usage_mb": 512,
    "max_cpu_usage_percent": 80,
}

# Test data generation constants
RANDOM_SEED = 42
DEFAULT_DATA_POINTS = 100
DEFAULT_VOLATILITY_SAMPLES = 252  # trading days in a year

# Test account values
TEST_ACCOUNT_EQUITY = Decimal("10000.0")
TEST_ACCOUNT_BALANCE = Decimal("9500.0")
TEST_MARGIN_USED = Decimal("500.0")
TEST_FREE_MARGIN = Decimal("9000.0")

# Position test values
TEST_POSITION_SIZE = Decimal("1.0")
TEST_ENTRY_PRICE = Decimal("50000.0")
TEST_MARK_PRICE = Decimal("51000.0")
TEST_UNREALIZED_PNL = Decimal("1000.0")
TEST_REALIZED_PNL = Decimal("500.0")

# Order book test values
DEFAULT_ORDER_BOOK_LEVELS = 5
DEFAULT_ORDER_BOOK_SPREAD = Decimal("0.5")
DEFAULT_ORDER_BOOK_STEP = Decimal("0.25")
DEFAULT_ORDER_BOOK_SIZE = Decimal("1.0")

# Test strategy parameters
STRATEGY_PARAMS = {
    "trend_following": {
        "lookback_period": 20,
        "entry_threshold": 0.02,
        "exit_threshold": -0.01,
        "stop_loss": 0.05,
        "take_profit": 0.10,
    },
    "mean_reversion": {
        "lookback_period": 30,
        "z_score_entry": 2.0,
        "z_score_exit": 0.5,
        "stop_loss": 0.03,
        "take_profit": 0.05,
    },
    "momentum": {
        "lookback_period": 10,
        "momentum_threshold": 0.03,
        "holding_period": 5,
        "stop_loss": 0.04,
        "take_profit": 0.08,
    },
}

# Test signal values
DEFAULT_SIGNAL_CONFIDENCE = 0.8
DEFAULT_STOP_LOSS_PCT = 0.05
DEFAULT_TAKE_PROFIT_PCT = 0.10

# Test monitoring values
DEFAULT_HEALTH_CHECK_INTERVAL = 60  # seconds
DEFAULT_METRIC_COLLECTION_INTERVAL = 30  # seconds
DEFAULT_ALERT_COOLDOWN = 300  # seconds

# Test validation constants
MAX_PRICE_DEVIATION = 0.10  # 10%
MAX_QUANTITY_DEVIATION = 0.05  # 5%
MIN_ORDER_SIZE = Decimal("0.001")
MAX_ORDER_SIZE = Decimal("100.0")
