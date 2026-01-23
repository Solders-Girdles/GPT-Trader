from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gpt_trader.preflight.core import PreflightCheck


def check_dependencies(checker: PreflightCheck) -> bool:
    """Ensure required Python packages are installed."""
    checker.section_header("2. DEPENDENCY CHECK")

    required_packages = [
        ("gpt_trader", "project import"),
        ("requests", "Coinbase REST client"),
        ("yaml", "YAML config loading (pyyaml)"),
        ("jwt", "Coinbase JWT auth (pyjwt)"),
        ("cryptography", "ES256 signing + encrypted secrets (cryptography)"),
    ]

    # Optional extras used by live trading and observability.
    optional_packages: list[tuple[str, str]] = [
        ("aiohttp", "webhook notifications + async tooling (extra: live-trade)"),
        ("websocket", "Coinbase WebSocket client (extra: live-trade)"),
        ("psutil", "system telemetry (extra: monitoring)"),
    ]

    # If we're running a non-dev preflight AND remote checks are enabled, the
    # runtime is expected to have live-trade extras installed.
    enforce_live_trade = (
        checker.context.profile not in {"dev", "demo"}
        and not checker.context.should_skip_remote_checks()
    )
    if enforce_live_trade:
        required_packages.extend(
            [
                ("aiohttp", "webhook notifications + async tooling (extra: live-trade)"),
                ("websocket", "Coinbase WebSocket client (extra: live-trade)"),
            ]
        )
        optional_packages = [
            pkg for pkg in optional_packages if pkg[0] not in {"aiohttp", "websocket"}
        ]

    all_good = True
    for package, reason in required_packages:
        try:
            if package == "gpt_trader":
                from gpt_trader import cli  # noqa: F401  # ensure our package import works
            else:
                __import__(package)
            checker.log_info(f"Package {package} found ({reason})")
        except ImportError:
            checker.log_error(f"Missing required package: {package} ({reason})")
            all_good = False

    for package, reason in optional_packages:
        try:
            __import__(package)
            checker.log_info(f"Optional package {package} found ({reason})")
        except ImportError:
            checker.log_warning(f"Optional package missing: {package} ({reason})")

    if all_good:
        checker.log_success("All required packages installed")
    return all_good
