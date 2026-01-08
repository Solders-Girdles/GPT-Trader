from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from gpt_trader.preflight.core import PreflightCheck


def check_risk_configuration(checker: PreflightCheck) -> bool:
    """Validate risk management configuration."""
    checker.section_header("6. RISK MANAGEMENT VALIDATION")

    try:
        from gpt_trader.features.live_trade.risk.config import RiskConfig

        config = RiskConfig.from_env()

        # Validate daily_loss_limit_pct (percentage-based, preferred)
        # Fall back to daily_loss_limit (absolute) if pct not configured
        has_pct_limit = config.daily_loss_limit_pct > 0
        has_abs_limit = config.daily_loss_limit > 0

        checks: list[tuple[str, object, Callable[[Any], bool]]] = [
            ("Max leverage", config.max_leverage, lambda x: 1 <= x <= 10),
            ("Liquidation buffer", config.min_liquidation_buffer_pct, lambda x: x >= 0.10),
            ("Position limit", config.max_position_pct_per_symbol, lambda x: 0 < x <= 0.25),
            ("Slippage guard", config.slippage_guard_bps, lambda x: 10 <= x <= 100),
        ]

        # Add daily loss limit check (prefer pct, fall back to absolute)
        if has_pct_limit:
            checks.append(
                (
                    "Daily loss limit (pct)",
                    config.daily_loss_limit_pct,
                    lambda x: 0 < x <= 0.20,  # 0-20% is reasonable
                )
            )
        elif has_abs_limit:
            checks.append(("Daily loss limit ($)", config.daily_loss_limit, lambda x: x > 0))
        else:
            checker.log_warning(
                "No daily loss limit configured - consider setting RISK_DAILY_LOSS_LIMIT_PCT"
            )

        all_good = True
        for name, value, validator in checks:
            if validator(value):
                checker.log_success(f"{name}: {value} ✓")
            else:
                checker.log_error(f"{name}: {value} - UNSAFE VALUE")
                all_good = False

        if config.kill_switch_enabled:
            checker.log_warning("Kill switch is ENABLED - all trading blocked")

        if config.reduce_only_mode:
            checker.log_warning("Reduce-only mode ENABLED - can only close positions")

        if config.daily_loss_limit_pct > 0.10:
            checker.log_warning(
                f"Daily loss limit {config.daily_loss_limit_pct:.0%} seems high for testing"
            )

        if config.max_leverage > 5:
            checker.log_warning(f"Leverage {config.max_leverage}x is aggressive")

        return all_good

    except Exception as exc:
        checker.log_error(f"Failed to validate risk config: {exc}")
        return False
