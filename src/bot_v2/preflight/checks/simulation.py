from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bot_v2.preflight.core import PreflightCheck


def simulate_dry_run(checker: "PreflightCheck") -> bool:
    """Simulate a dry-run execution to ensure components wire together."""
    checker.section_header("11. DRY-RUN SIMULATION")

    print("Simulating dry-run execution...")

    try:
        from bot_v2.orchestration.bootstrap import build_bot
        from bot_v2.orchestration.broker_factory import create_brokerage
        from bot_v2.orchestration.configuration import BotConfig
        from bot_v2.features.live_trade.strategies.perps_baseline import (
            BaselinePerpsStrategy,
        )

        config = BotConfig.from_profile(profile=checker.profile, dry_run=True, mock_broker=True)
        checker.log_info(f"Config: {checker.profile} profile, dry_run=True, deterministic broker")

        try:
            broker, *_ = create_brokerage()
            if broker:
                checker.log_success("Broker factory initialized")
        except Exception as exc:
            checker.log_warning(f"Broker initialization warning: {exc}")

        strategy = BaselinePerpsStrategy()
        checker.log_success("Strategy initialized")

        bot, registry = build_bot(config)
        checker.log_success("PerpsBot constructed via bootstrap")
        if bot.broker:
            checker.log_success("Broker instance available")
        if bot.risk_manager:
            checker.log_success("Risk manager initialized")
        checker.log_success("Dry-run simulation passed")
        return True

    except Exception as exc:
        checker.log_error(f"Dry-run simulation failed: {exc}")
        return False
