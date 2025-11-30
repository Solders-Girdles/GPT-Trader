"""
Configuration validator for BotConfig.
"""

from gpt_trader.orchestration.configuration.bot_config.bot_config import BotConfig


class ConfigValidationError(Exception):
    """Raised when configuration validation fails."""

    pass


def validate_config(config: BotConfig) -> list[str]:
    """
    Validate the configuration and return a list of error messages.
    Returns an empty list if the configuration is valid.
    """
    errors = []

    # Validate Risk Config
    risk = config.risk
    if risk.position_fraction <= 0 or risk.position_fraction > 1:
        errors.append(
            f"risk.position_fraction must be between 0 and 1, got {risk.position_fraction}"
        )

    if risk.stop_loss_pct <= 0 or risk.stop_loss_pct >= 1:
        errors.append(f"risk.stop_loss_pct must be between 0 and 1, got {risk.stop_loss_pct}")

    if risk.take_profit_pct <= 0:
        errors.append(f"risk.take_profit_pct must be positive, got {risk.take_profit_pct}")

    if risk.max_leverage < 1:
        errors.append(f"risk.max_leverage must be >= 1, got {risk.max_leverage}")

    # Validate Strategy Config
    strategy = config.strategy
    # Check if it's the expected strategy type before accessing fields
    if hasattr(strategy, "short_ma_period") and hasattr(strategy, "long_ma_period"):
        if strategy.short_ma_period >= strategy.long_ma_period:
            errors.append(
                f"strategy.short_ma_period ({strategy.short_ma_period}) must be less than "
                f"strategy.long_ma_period ({strategy.long_ma_period})"
            )

    # Validate Symbols
    if not config.symbols:
        errors.append("symbols list cannot be empty")

    return errors
