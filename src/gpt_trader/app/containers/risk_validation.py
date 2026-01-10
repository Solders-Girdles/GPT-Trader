"""Risk and validation sub-container for ApplicationContainer.

This container manages risk and validation-related dependencies:
- LiveRiskManager (leverage limits, loss limits, exposure caps)
- ValidationFailureTracker (consecutive failure tracking with escalation)
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from gpt_trader.app.config import BotConfig

if TYPE_CHECKING:
    from gpt_trader.features.live_trade.execution.validation import ValidationFailureTracker
    from gpt_trader.features.live_trade.risk.manager import LiveRiskManager
    from gpt_trader.persistence.event_store import EventStore


class RiskValidationContainer:
    """Container for risk and validation-related dependencies.

    Lazily initializes LiveRiskManager and ValidationFailureTracker.

    Args:
        config: Bot configuration.
        event_store_provider: Callable returning the EventStore instance.
            This is a callable (not the instance) to support lazy resolution
            and avoid initialization order issues.
    """

    def __init__(
        self,
        config: BotConfig,
        event_store_provider: Callable[[], EventStore],
    ):
        self._config = config
        self._event_store_provider = event_store_provider

        self._risk_manager: LiveRiskManager | None = None
        self._validation_failure_tracker: ValidationFailureTracker | None = None

    @property
    def risk_manager(self) -> LiveRiskManager:
        """Get or create the risk manager instance.

        Creates a LiveRiskManager configured from BotConfig.risk settings.
        The manager enforces leverage limits, daily loss limits, position
        exposure caps, and other risk controls.
        """
        if self._risk_manager is None:
            from decimal import Decimal

            from gpt_trader.features.live_trade.risk.config import RiskConfig
            from gpt_trader.features.live_trade.risk.manager import LiveRiskManager

            # Adapt BotConfig.risk (BotRiskConfig) to RiskConfig
            bot_risk = self._config.risk

            # Derive kill_switch_enabled from active strategy config
            if self._config.strategy_type == "mean_reversion":
                kill_switch = self._config.mean_reversion.kill_switch_enabled
            else:
                # baseline, ensemble, or any other type uses strategy config
                kill_switch = getattr(self._config.strategy, "kill_switch_enabled", False)

            risk_config = RiskConfig(
                max_leverage=bot_risk.max_leverage,
                # daily_loss_limit is absolute dollar amount (legacy)
                daily_loss_limit=Decimal("100"),
                # daily_loss_limit_pct is percentage of equity (used by LiveRiskManager)
                daily_loss_limit_pct=bot_risk.daily_loss_limit_pct,
                max_position_pct_per_symbol=float(bot_risk.position_fraction),
                # Map other relevant fields
                kill_switch_enabled=kill_switch,
                reduce_only_mode=self._config.reduce_only_mode,
            )

            self._risk_manager = LiveRiskManager(
                config=risk_config,
                event_store=self._event_store_provider(),
            )
        return self._risk_manager

    @property
    def validation_failure_tracker(self) -> ValidationFailureTracker:
        """Get or create the validation failure tracker.

        The tracker monitors consecutive validation failures and can trigger
        escalation (e.g., reduce-only mode) when thresholds are exceeded.

        Note: The escalation callback is not set here - it should be configured
        by the caller (e.g., TradingEngine) who knows the escalation target.
        """
        if self._validation_failure_tracker is None:
            from gpt_trader.features.live_trade.execution.validation import (
                ValidationFailureTracker as VFT,
            )

            self._validation_failure_tracker = VFT()
        return self._validation_failure_tracker

    def reset_risk_manager(self) -> None:
        """Reset the risk manager, forcing re-creation on next access."""
        self._risk_manager = None

    def reset_validation_failure_tracker(self) -> None:
        """Reset the validation failure tracker, forcing re-creation on next access."""
        self._validation_failure_tracker = None
