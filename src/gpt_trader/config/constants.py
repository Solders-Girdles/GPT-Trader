"""
Centralized configuration constants.

These values can be overridden via environment variables where noted.
Moving hardcoded values here makes them easier to tune without code changes.
"""

import os

# =============================================================================
# HTTP/API Configuration
# =============================================================================

# Default HTTP request timeout in seconds
DEFAULT_HTTP_TIMEOUT: int = int(os.getenv("GPT_TRADER_HTTP_TIMEOUT", "30"))

# Webhook notification timeout in seconds
WEBHOOK_TIMEOUT: int = int(os.getenv("GPT_TRADER_WEBHOOK_TIMEOUT", "5"))

# =============================================================================
# Retry Configuration
# =============================================================================

# Base delay for exponential backoff (seconds)
RETRY_BASE_DELAY: float = float(os.getenv("GPT_TRADER_RETRY_BASE_DELAY", "0.5"))

# Backoff multiplier (delay = base * (multiplier ** attempt))
RETRY_BACKOFF_MULTIPLIER: float = float(os.getenv("GPT_TRADER_RETRY_BACKOFF_MULTIPLIER", "2.0"))

# Maximum retry attempts for HTTP requests
MAX_HTTP_RETRIES: int = int(os.getenv("GPT_TRADER_MAX_HTTP_RETRIES", "3"))

# =============================================================================
# WebSocket Configuration
# =============================================================================

# WebSocket thread join timeout in seconds
WS_JOIN_TIMEOUT: float = float(os.getenv("GPT_TRADER_WS_JOIN_TIMEOUT", "2.0"))

# -----------------------------------------------------------------------------
# WebSocket Reconnection with Exponential Backoff + Jitter
# -----------------------------------------------------------------------------

# Base delay for exponential backoff (seconds)
WS_RECONNECT_BACKOFF_BASE_SECONDS: float = float(
    os.getenv("GPT_TRADER_WS_RECONNECT_BACKOFF_BASE", "2.0")
)

# Maximum reconnection delay (caps exponential growth)
WS_RECONNECT_BACKOFF_MAX_SECONDS: float = float(
    os.getenv("GPT_TRADER_WS_RECONNECT_BACKOFF_MAX", "60.0")
)

# Backoff multiplier (delay = base * (multiplier ** attempt))
WS_RECONNECT_BACKOFF_MULTIPLIER: float = float(
    os.getenv("GPT_TRADER_WS_RECONNECT_BACKOFF_MULTIPLIER", "2.0")
)

# Jitter percentage (0.25 = ±25% randomization to prevent thundering herd)
WS_RECONNECT_JITTER_PCT: float = float(os.getenv("GPT_TRADER_WS_RECONNECT_JITTER_PCT", "0.25"))

# Maximum reconnection attempts before triggering degradation (0 = unlimited)
WS_RECONNECT_MAX_ATTEMPTS: int = int(os.getenv("GPT_TRADER_WS_RECONNECT_MAX_ATTEMPTS", "10"))

# Seconds of stable connection required to reset attempt counter
WS_RECONNECT_RESET_SECONDS: float = float(
    os.getenv("GPT_TRADER_WS_RECONNECT_RESET_SECONDS", "60.0")
)

# Pause duration when max attempts exceeded (triggers DegradationState.pause_all)
WS_RECONNECT_PAUSE_SECONDS: int = int(os.getenv("GPT_TRADER_WS_RECONNECT_PAUSE_SECONDS", "300"))

# Legacy aliases for backward compatibility
WS_RECONNECT_DELAY: int = int(WS_RECONNECT_BACKOFF_BASE_SECONDS)
MAX_WS_RECONNECT_DELAY_SECONDS: float = WS_RECONNECT_BACKOFF_MAX_SECONDS
MAX_WS_RECONNECT_ATTEMPTS: int = WS_RECONNECT_MAX_ATTEMPTS

# =============================================================================
# Security / Validation Configuration
# =============================================================================

# Symbols blocked from trading (comma-separated in env var)
_blocked_symbols_env = os.getenv("GPT_TRADER_BLOCKED_SYMBOLS", "TEST,DEBUG,HACK")
BLOCKED_SYMBOLS: frozenset[str] = frozenset(
    s.strip().upper() for s in _blocked_symbols_env.split(",") if s.strip()
)

# =============================================================================
# Market Hours Configuration
# =============================================================================

# US equity market hours (Eastern Time)
MARKET_OPEN_HOUR: int = int(os.getenv("GPT_TRADER_MARKET_OPEN_HOUR", "9"))
MARKET_CLOSE_HOUR: int = int(os.getenv("GPT_TRADER_MARKET_CLOSE_HOUR", "16"))

# =============================================================================
# Rate Limiting Configuration
# =============================================================================

# Default rate limit (requests per minute)
DEFAULT_RATE_LIMIT_PER_MINUTE: int = int(os.getenv("GPT_TRADER_RATE_LIMIT_PER_MINUTE", "100"))

# Rate limit warning threshold (percentage of limit)
RATE_LIMIT_WARNING_THRESHOLD: float = float(os.getenv("GPT_TRADER_RATE_LIMIT_WARNING_PCT", "0.8"))

# Rate limit window in seconds (1 minute)
RATE_LIMIT_WINDOW_SECONDS: int = int(os.getenv("GPT_TRADER_RATE_LIMIT_WINDOW", "60"))

# =============================================================================
# Database Configuration
# =============================================================================

# SQLite busy timeout in milliseconds
SQLITE_BUSY_TIMEOUT_MS: int = int(os.getenv("GPT_TRADER_SQLITE_BUSY_TIMEOUT", "5000"))

# Maximum number of database integrity issues to log before suppressing
MAX_INTEGRITY_ISSUES_TO_LOG: int = int(os.getenv("GPT_TRADER_MAX_INTEGRITY_LOG", "5"))

# =============================================================================
# Health & Monitoring Configuration
# =============================================================================

# Health check request read timeout in seconds
HEALTH_CHECK_READ_TIMEOUT_SECONDS: float = float(os.getenv("GPT_TRADER_HEALTH_READ_TIMEOUT", "5.0"))

# External heartbeat ping timeout in seconds
HEARTBEAT_PING_TIMEOUT_SECONDS: int = int(os.getenv("GPT_TRADER_HEARTBEAT_PING_TIMEOUT", "10"))

# Heartbeat health multiplier (healthy if last beat within interval * multiplier)
HEARTBEAT_HEALTH_MULTIPLIER: int = int(os.getenv("GPT_TRADER_HEARTBEAT_HEALTH_MULT", "2"))

# Default guard state cache interval in seconds
DEFAULT_GUARD_CACHE_INTERVAL_SECONDS: float = float(
    os.getenv("GPT_TRADER_GUARD_CACHE_INTERVAL", "60.0")
)

# Health check runner interval in seconds
HEALTH_CHECK_INTERVAL_SECONDS: float = float(os.getenv("GPT_TRADER_HEALTH_CHECK_INTERVAL", "30.0"))

# =============================================================================
# Trading Calculation Defaults
# =============================================================================

# Default quote increment for price calculations
DEFAULT_QUOTE_INCREMENT: float = float(os.getenv("GPT_TRADER_DEFAULT_QUOTE_INCREMENT", "0.01"))

# Minimum collateral change to trigger logging (in dollars)
MIN_COLLATERAL_CHANGE_TO_LOG: float = float(os.getenv("GPT_TRADER_MIN_COLLATERAL_LOG", "0.01"))

# Default volatility window periods for risk calculations
DEFAULT_VOLATILITY_WINDOW_PERIODS: int = int(os.getenv("GPT_TRADER_VOLATILITY_WINDOW", "20"))

# Minimum volatility window threshold
MIN_VOLATILITY_WINDOW_THRESHOLD: int = int(os.getenv("GPT_TRADER_MIN_VOLATILITY_WINDOW", "5"))

# =============================================================================
# API Resilience Feature Flags
# =============================================================================

# Response caching - reduces API calls for stable data
CACHE_ENABLED: bool = os.getenv("GPT_TRADER_CACHE_ENABLED", "true").lower() == "true"
CACHE_DEFAULT_TTL: float = float(os.getenv("GPT_TRADER_CACHE_TTL", "30.0"))
CACHE_MAX_SIZE: int = int(os.getenv("GPT_TRADER_CACHE_MAX_SIZE", "1000"))

# Circuit breaker - prevents hammering failing endpoints
CIRCUIT_BREAKER_ENABLED: bool = (
    os.getenv("GPT_TRADER_CIRCUIT_BREAKER_ENABLED", "true").lower() == "true"
)
CIRCUIT_FAILURE_THRESHOLD: int = int(os.getenv("GPT_TRADER_CIRCUIT_FAILURE_THRESHOLD", "5"))
CIRCUIT_RECOVERY_TIMEOUT: float = float(os.getenv("GPT_TRADER_CIRCUIT_RECOVERY_TIMEOUT", "30.0"))
CIRCUIT_SUCCESS_THRESHOLD: int = int(os.getenv("GPT_TRADER_CIRCUIT_SUCCESS_THRESHOLD", "2"))

# API metrics collection - tracks latency and error rates
METRICS_ENABLED: bool = os.getenv("GPT_TRADER_METRICS_ENABLED", "true").lower() == "true"
METRICS_HISTORY_SIZE: int = int(os.getenv("GPT_TRADER_METRICS_HISTORY", "100"))

# Request priority - prioritizes critical requests under rate limit pressure
PRIORITY_ENABLED: bool = os.getenv("GPT_TRADER_PRIORITY_ENABLED", "true").lower() == "true"
PRIORITY_THRESHOLD_HIGH: float = float(os.getenv("GPT_TRADER_PRIORITY_THRESHOLD_HIGH", "0.70"))
PRIORITY_THRESHOLD_CRITICAL: float = float(
    os.getenv("GPT_TRADER_PRIORITY_THRESHOLD_CRITICAL", "0.85")
)

# Adaptive throttling - proactive pacing instead of reactive blocking
ADAPTIVE_THROTTLE_ENABLED: bool = (
    os.getenv("GPT_TRADER_ADAPTIVE_THROTTLE", "true").lower() == "true"
)
THROTTLE_TARGET_UTILIZATION: float = float(os.getenv("GPT_TRADER_THROTTLE_TARGET", "0.7"))

# =============================================================================
# Observability Configuration
# =============================================================================

# Prometheus metrics endpoint (served on health server)
METRICS_ENDPOINT_ENABLED: bool = os.getenv("GPT_TRADER_METRICS_ENDPOINT_ENABLED", "0").lower() in (
    "1",
    "true",
    "yes",
)

# OpenTelemetry tracing
OTEL_ENABLED: bool = os.getenv("GPT_TRADER_OTEL_ENABLED", "0").lower() in ("1", "true", "yes")
OTEL_SERVICE_NAME: str = os.getenv("OTEL_SERVICE_NAME", "gpt-trader")
OTEL_EXPORTER_ENDPOINT: str | None = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
