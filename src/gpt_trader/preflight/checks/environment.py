from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gpt_trader.preflight.core import PreflightCheck


def check_environment_variables(checker: PreflightCheck) -> bool:
    """Validate critical environment configuration and credential presence."""
    checker.section_header("3. ENVIRONMENT CONFIGURATION")

    ctx = checker.context
    all_good = True
    for var, (expected, strict) in ctx.expected_env_defaults().items():
        actual = os.getenv(var)
        if actual == expected:
            checker.log_info(f"{var}={expected}")
            continue

        if actual is None:
            message = f"{var} not set, expected '{expected}'"
        else:
            message = f"{var}={actual}, expected '{expected}'"

        if strict:
            checker.log_error(message)
            all_good = False
        else:
            checker.log_warning(message)

    api_key, private_key = ctx.resolve_cdp_credentials()
    if not api_key:
        if ctx.should_skip_remote_checks():
            checker.log_warning(
                "CDP API key not configured; remote connectivity checks will be skipped"
            )
        else:
            checker.log_error("CDP API key not found (COINBASE_PROD_CDP_API_KEY)")
            all_good = False
    else:
        if api_key.startswith("organizations/") and "/apiKeys/" in api_key:
            checker.log_success(f"CDP API key format valid: {api_key[:30]}...")
        else:
            message = "Invalid CDP API key format. Expected: organizations/.../apiKeys/..."
            if ctx.should_skip_remote_checks():
                checker.log_warning(message)
            else:
                checker.log_error(message)
                all_good = False

    if not private_key:
        if ctx.should_skip_remote_checks():
            checker.log_warning(
                "CDP private key not configured; remote connectivity checks will be skipped"
            )
        else:
            checker.log_error("CDP private key not found (COINBASE_PROD_CDP_PRIVATE_KEY)")
            all_good = False
    else:
        if "BEGIN EC PRIVATE KEY" in private_key:
            checker.log_success("CDP private key found (EC format)")
        else:
            message = "Invalid private key format (must be EC private key)"
            if ctx.should_skip_remote_checks():
                checker.log_warning(message)
            else:
                checker.log_error(message)
                all_good = False

    risk_vars = {
        "RISK_MAX_LEVERAGE": (1, 10),
        "RISK_DAILY_LOSS_LIMIT": (10, 10000),
        "RISK_MAX_POSITION_PCT_PER_SYMBOL": (0.01, 0.5),
    }

    for var, (min_val, max_val) in risk_vars.items():
        value = os.getenv(var)
        if value:
            try:
                num = float(value)
                if min_val <= num <= max_val:
                    checker.log_info(f"{var}={value} (within safe range)")
                else:
                    checker.log_warning(
                        f"{var}={value} outside recommended range [{min_val}, {max_val}]"
                    )
            except ValueError:
                checker.log_error(f"{var}={value} is not a valid number")
                all_good = False
        else:
            checker.log_warning(f"{var} not set, using defaults")

    return all_good
