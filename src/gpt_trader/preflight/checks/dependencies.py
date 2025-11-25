from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gpt_trader.preflight.core import PreflightCheck


def check_dependencies(checker: PreflightCheck) -> bool:
    """Ensure required Python packages are installed."""
    checker.section_header("2. DEPENDENCY CHECK")

    required_packages = [
        "gpt_trader",
        "decimal",
        "pytest",
        "cryptography",
        "jwt",
        "websockets",
        "aiohttp",
    ]

    all_good = True
    for package in required_packages:
        try:
            if package == "gpt_trader":
                from gpt_trader import cli  # noqa: F401  # ensure our package import works
            else:
                __import__(package)
            checker.log_info(f"Package {package} found")
        except ImportError:
            checker.log_error(f"Missing required package: {package}")
            all_good = False

    if all_good:
        checker.log_success("All required packages installed")
    return all_good
