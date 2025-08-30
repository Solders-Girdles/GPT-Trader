"""Alpaca Configuration Template.

This file provides configuration templates and utilities for setting up
Alpaca paper trading integration with the GPT-Trader system.

Copy this file to alpaca_config.py and customize for your setup.
"""

import os
from dataclasses import dataclass
from typing import List, Optional

from bot.brokers.alpaca import AlpacaConfig, MarketDataConfig, PaperTradingConfig


# ============================================================================
# Environment Variables Template
# ============================================================================

REQUIRED_ENV_VARS = """
# Alpaca API Credentials (get from https://alpaca.markets)
export ALPACA_API_KEY="your_api_key_here"
export ALPACA_SECRET_KEY="your_secret_key_here"

# Optional: Paper trading mode (defaults to true)
export ALPACA_PAPER="true"

# Optional: Base URL override
# export ALPACA_BASE_URL="https://paper-api.alpaca.markets"
"""


# ============================================================================
# Configuration Presets
# ============================================================================

def get_development_config() -> PaperTradingConfig:
    """Get configuration for development/testing."""
    alpaca_config = AlpacaConfig(
        api_key=os.getenv("ALPACA_API_KEY", ""),
        secret_key=os.getenv("ALPACA_SECRET_KEY", ""),
        paper_trading=True,  # Always use paper trading for development
        max_retries=3,
        retry_delay=1.0,
        rate_limit_delay=0.2,  # More aggressive rate limiting for development
    )
    
    return PaperTradingConfig(
        alpaca_config=alpaca_config,
        enable_real_time_data=True,
        data_symbols=["AAPL", "MSFT", "GOOGL", "SPY"],  # Common symbols for testing
        simulate_execution_delay=True,
        min_execution_delay_ms=50,
        max_execution_delay_ms=200,
        max_order_value=1000.0,  # Low limit for safety
        max_daily_trades=50,  # Conservative limit
        log_all_orders=True,
        save_execution_log=True,
    )


def get_testing_config() -> PaperTradingConfig:
    """Get configuration for automated testing."""
    alpaca_config = AlpacaConfig(
        api_key=os.getenv("ALPACA_API_KEY", ""),
        secret_key=os.getenv("ALPACA_SECRET_KEY", ""),
        paper_trading=True,
        max_retries=2,
        retry_delay=0.5,
        rate_limit_delay=0.1,
    )
    
    return PaperTradingConfig(
        alpaca_config=alpaca_config,
        enable_real_time_data=False,  # Disable for faster tests
        data_symbols=[],
        simulate_execution_delay=False,  # No delays for tests
        max_order_value=100.0,  # Very low limit
        max_daily_trades=10,
        log_all_orders=False,  # Reduce noise in tests
        save_execution_log=False,
    )


def get_production_config() -> PaperTradingConfig:
    """Get configuration for production paper trading."""
    alpaca_config = AlpacaConfig(
        api_key=os.getenv("ALPACA_API_KEY", ""),
        secret_key=os.getenv("ALPACA_SECRET_KEY", ""),
        paper_trading=True,  # Keep as paper trading for safety
        max_retries=5,
        retry_delay=2.0,
        rate_limit_delay=0.05,  # Less aggressive for production
    )
    
    return PaperTradingConfig(
        alpaca_config=alpaca_config,
        enable_real_time_data=True,
        data_symbols=[
            # Large cap stocks
            "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA",
            # ETFs  
            "SPY", "QQQ", "IWM", "VTI", "VOO",
            # Add your strategy symbols here
        ],
        simulate_execution_delay=True,
        min_execution_delay_ms=100,
        max_execution_delay_ms=500,
        max_order_value=50000.0,  # Higher limit for production
        max_daily_trades=200,
        log_all_orders=True,
        save_execution_log=True,
    )


# ============================================================================
# Custom Configuration Builder
# ============================================================================

@dataclass
class AlpacaConfigBuilder:
    """Builder for custom Alpaca configurations."""
    
    # API settings
    api_key: str = ""
    secret_key: str = ""
    paper_trading: bool = True
    base_url: Optional[str] = None
    
    # Connection settings
    max_retries: int = 3
    retry_delay: float = 1.0
    rate_limit_delay: float = 0.1
    
    # Data feed settings
    enable_real_time_data: bool = True
    data_symbols: List[str] = None
    subscribe_quotes: bool = True
    subscribe_trades: bool = True
    subscribe_bars: bool = False
    
    # Execution settings
    simulate_execution_delay: bool = True
    min_execution_delay_ms: int = 100
    max_execution_delay_ms: int = 500
    
    # Risk limits
    max_order_value: float = 10000.0
    max_daily_trades: int = 100
    
    # Logging
    log_all_orders: bool = True
    save_execution_log: bool = True
    
    def __post_init__(self):
        """Initialize defaults."""
        if self.data_symbols is None:
            self.data_symbols = ["AAPL", "MSFT", "SPY"]
        
        # Load from environment if not set
        if not self.api_key:
            self.api_key = os.getenv("ALPACA_API_KEY", "")
        if not self.secret_key:
            self.secret_key = os.getenv("ALPACA_SECRET_KEY", "")
    
    def build(self) -> PaperTradingConfig:
        """Build the final configuration."""
        if not self.api_key or not self.secret_key:
            raise ValueError("API key and secret key are required")
        
        alpaca_config = AlpacaConfig(
            api_key=self.api_key,
            secret_key=self.secret_key,
            paper_trading=self.paper_trading,
            base_url=self.base_url,
            max_retries=self.max_retries,
            retry_delay=self.retry_delay,
            rate_limit_delay=self.rate_limit_delay,
        )
        
        return PaperTradingConfig(
            alpaca_config=alpaca_config,
            enable_real_time_data=self.enable_real_time_data,
            data_symbols=self.data_symbols.copy(),
            simulate_execution_delay=self.simulate_execution_delay,
            min_execution_delay_ms=self.min_execution_delay_ms,
            max_execution_delay_ms=self.max_execution_delay_ms,
            max_order_value=self.max_order_value,
            max_daily_trades=self.max_daily_trades,
            log_all_orders=self.log_all_orders,
            save_execution_log=self.save_execution_log,
        )


# ============================================================================
# Configuration Validation
# ============================================================================

def validate_config(config: PaperTradingConfig) -> List[str]:
    """Validate configuration and return list of issues."""
    issues = []
    
    # Check API credentials
    if not config.alpaca_config.api_key:
        issues.append("ALPACA_API_KEY is required")
    if not config.alpaca_config.secret_key:
        issues.append("ALPACA_SECRET_KEY is required")
    
    # Check risk limits
    if config.max_order_value <= 0:
        issues.append("max_order_value must be positive")
    if config.max_daily_trades <= 0:
        issues.append("max_daily_trades must be positive")
    
    # Check execution delays
    if config.simulate_execution_delay:
        if config.min_execution_delay_ms < 0:
            issues.append("min_execution_delay_ms cannot be negative")
        if config.max_execution_delay_ms < config.min_execution_delay_ms:
            issues.append("max_execution_delay_ms must be >= min_execution_delay_ms")
    
    # Check data symbols
    if config.enable_real_time_data and not config.data_symbols:
        issues.append("data_symbols required when real-time data is enabled")
    
    return issues


def check_environment() -> List[str]:
    """Check if environment is properly configured."""
    issues = []
    
    required_vars = ["ALPACA_API_KEY", "ALPACA_SECRET_KEY"]
    for var in required_vars:
        if not os.getenv(var):
            issues.append(f"Environment variable {var} is not set")
    
    return issues


# ============================================================================
# Example Usage
# ============================================================================

def example_usage():
    """Example of how to use the configuration builder."""
    
    # Method 1: Use presets
    dev_config = get_development_config()
    test_config = get_testing_config()
    prod_config = get_production_config()
    
    # Method 2: Use builder for custom config
    builder = AlpacaConfigBuilder()
    builder.data_symbols = ["AAPL", "TSLA", "NVDA"]
    builder.max_order_value = 5000.0
    builder.simulate_execution_delay = False
    custom_config = builder.build()
    
    # Method 3: Validate configuration
    issues = validate_config(custom_config)
    if issues:
        print("Configuration issues:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("Configuration is valid")
    
    # Method 4: Check environment
    env_issues = check_environment()
    if env_issues:
        print("Environment issues:")
        for issue in env_issues:
            print(f"  - {issue}")
    else:
        print("Environment is properly configured")


if __name__ == "__main__":
    print("Alpaca Configuration Template")
    print("=" * 40)
    print("\nTo set up Alpaca integration:")
    print("1. Get API keys from https://alpaca.markets")
    print("2. Set environment variables:")
    print(REQUIRED_ENV_VARS)
    print("3. Choose a configuration preset or build custom")
    print("4. Validate configuration before use")
    print("\nRunning example...")
    example_usage()