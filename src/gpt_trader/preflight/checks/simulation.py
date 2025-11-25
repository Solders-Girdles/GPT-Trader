from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gpt_trader.preflight.core import PreflightCheck


def simulate_dry_run(checker: PreflightCheck) -> bool:
    """Simulate a dry-run execution to ensure components wire together."""
    checker.section_header("11. DRY-RUN SIMULATION")

    print("Simulating dry-run execution...")

    try:
        from gpt_trader.app.container import create_application_container
        from gpt_trader.orchestration.configuration import BotConfig

        config = BotConfig.from_profile(profile=checker.profile, dry_run=True, mock_broker=True)
        checker.log_info(f"Config: {checker.profile} profile, dry_run=True, deterministic broker")

        container = create_application_container(config)
        checker.log_success("Application container initialized")

        bot = container.create_bot()
        checker.log_success("TradingBot constructed via container")

        if bot.engine:
            checker.log_success("Trading engine available")

        checker.log_success("Dry-run simulation passed")
        return True

    except Exception as exc:
        checker.log_error(f"Dry-run simulation failed: {exc}")
        return False
