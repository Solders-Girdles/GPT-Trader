"""Startup validation for GPT-Trader.

This module ensures all required configuration and secrets are available
before the application starts.
"""

import logging
import sys
from pathlib import Path

from bot.config.demo_mode import DemoModeConfig, setup_demo_mode
from bot.security.secrets_manager import ConfigurationError, get_secret_manager

logger = logging.getLogger(__name__)


def validate_startup(raise_on_failure: bool = True) -> bool:
    """Validate application startup requirements.

    This function:
    1. Validates all required secrets are available
    2. Checks for configuration issues
    3. Ensures necessary directories exist

    Args:
        raise_on_failure: If True, raises exception on validation failure.

    Returns:
        True if validation passes, False otherwise.

    Raises:
        ConfigurationError: If validation fails and raise_on_failure is True.
    """
    try:
        # Check for demo mode first
        if DemoModeConfig.is_demo_mode():
            setup_demo_mode()
            logger.info("Demo mode activated - using mock credentials")
            if sys.stdout.isatty():  # Only show in interactive terminals
                print(DemoModeConfig.get_demo_warning())

        # Validate secrets
        secret_manager = get_secret_manager()
        validation_results = secret_manager.validate_startup_secrets(
            raise_on_missing=raise_on_failure and not DemoModeConfig.is_demo_mode()
        )

        # Log validation results
        total_secrets = len(validation_results)
        valid_secrets = sum(1 for v in validation_results.values() if v)

        if valid_secrets < total_secrets:
            logger.warning(f"Secret validation: {valid_secrets}/{total_secrets} secrets found")
            missing = [k for k, v in validation_results.items() if not v]
            logger.warning(f"Missing secrets: {', '.join(missing)}")
        else:
            logger.info(f"All {total_secrets} secrets validated successfully")

        # Create necessary directories
        required_dirs = [
            Path("logs"),
            Path("data/cache"),
            Path("data/backtests"),
        ]

        for dir_path in required_dirs:
            if not dir_path.exists():
                dir_path.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created directory: {dir_path}")

        # Check for .env file (should not exist in production)
        env_file = Path(".env")
        if env_file.exists():
            logger.warning(
                "WARNING: .env file found in project root. "
                "This file should not exist in production. "
                "Use .env.local or environment variables instead."
            )

        logger.info("Startup validation completed successfully")
        return True

    except ConfigurationError as e:
        logger.error(f"Startup validation failed: {e}")
        if raise_on_failure:
            raise
        return False
    except Exception as e:
        logger.error(f"Unexpected error during startup validation: {e}")
        if raise_on_failure:
            raise ConfigurationError(f"Startup validation failed: {e}")
        return False


def initialize_application(config_path: Path | None = None) -> None:
    """Initialize the application with proper validation.

    This should be called at the beginning of any main entry point.

    Args:
        config_path: Optional path to configuration file.

    Raises:
        ConfigurationError: If initialization fails.
    """
    # Setup basic logging for startup
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    logger.info("=" * 60)
    logger.info("GPT-Trader Application Initialization")
    logger.info("=" * 60)

    # Validate startup requirements
    validate_startup(raise_on_failure=True)

    # Load configuration if provided
    if config_path and config_path.exists():
        logger.info(f"Loading configuration from: {config_path}")
        # Configuration loading would happen here

    logger.info("Application initialized successfully")
    logger.info("=" * 60)


if __name__ == "__main__":
    # Run validation when module is executed directly
    try:
        initialize_application()
        print("\n✅ All startup validations passed!")
    except ConfigurationError as e:
        print(f"\n❌ Startup validation failed: {e}")
        sys.exit(1)
