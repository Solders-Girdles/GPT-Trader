"""
Centralized configuration constants.

These values can be overridden via environment variables where noted.
Moving hardcoded values here makes them easier to tune without code changes.
"""

from decimal import Decimal
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

# WebSocket reconnection delay in seconds
WS_RECONNECT_DELAY: int = int(os.getenv("GPT_TRADER_WS_RECONNECT_DELAY", "5"))

# WebSocket thread join timeout in seconds
WS_JOIN_TIMEOUT: float = float(os.getenv("GPT_TRADER_WS_JOIN_TIMEOUT", "2.0"))

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
