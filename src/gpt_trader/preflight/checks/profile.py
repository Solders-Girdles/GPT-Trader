from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gpt_trader.preflight.core import PreflightCheck


def check_profile_configuration(checker: "PreflightCheck") -> bool:
    """Validate the selected trading profile configuration file."""
    checker.section_header("8. PROFILE CONFIGURATION")

    profile_path = Path(f"config/profiles/{checker.profile}.yaml")
    if profile_path.exists():
        checker.log_success(f"Profile '{checker.profile}' found at {profile_path}")

        try:
            import yaml

            with open(profile_path) as handle:
                profile_config = yaml.safe_load(handle)

            if checker.profile == "canary":
                expected = {
                    "trading.mode": "reduce_only",
                    "trading.position_sizing.max_position_size": 0.01,
                    "risk_management.daily_loss_limit": 10.00,
                    "risk_management.max_leverage": 1.0,
                }

                for key, expected_value in expected.items():
                    keys = key.split(".")
                    value = profile_config
                    for part in keys:
                        value = value.get(part, {})

                    if value == expected_value:
                        checker.log_info(f"{key} = {value} ✓")
                    else:
                        checker.log_warning(f"{key} = {value}, expected {expected_value}")

            checker.log_success(f"Profile '{checker.profile}' validated")
            return True

        except Exception as exc:
            checker.log_error(f"Failed to parse profile: {exc}")
            return False

    checker.log_warning(f"Profile '{checker.profile}' not found, will use defaults")
    if checker.profile == "canary":
        checker.log_info("Canary defaults: 0.01 BTC max, $10 daily loss, reduce-only")
    elif checker.profile == "prod":
        checker.log_warning("Production profile - ensure you've tested with canary first!")

    return True
