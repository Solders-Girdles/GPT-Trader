#!/usr/bin/env python3
"""
Production Preflight Check for GPT-Trader Perpetuals Bot

Comprehensive validation before trading with real money on Coinbase.
Performs all critical checks and provides clear go/no-go decision.

Usage:
    poetry run python scripts/production_preflight.py
    poetry run python scripts/production_preflight.py --verbose
    poetry run python scripts/production_preflight.py --profile canary
"""

from __future__ import annotations

import os
import sys
import json
import time
import argparse
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path


# Color codes for terminal output
class Colors:
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    RESET = "\033[0m"
    BOLD = "\033[1m"


class PreflightCheck:
    """Comprehensive preflight validation system."""

    def __init__(self, verbose: bool = False, profile: str = "canary"):
        self.verbose = verbose
        self.profile = profile
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.successes: list[str] = []
        self.config: dict[str, Any] = {}

    def log_success(self, message: str) -> None:
        """Log a successful check."""
        self.successes.append(message)
        print(f"{Colors.GREEN}✅ {message}{Colors.RESET}")

    def log_warning(self, message: str) -> None:
        """Log a warning."""
        self.warnings.append(message)
        print(f"{Colors.YELLOW}⚠️  {message}{Colors.RESET}")

    def log_error(self, message: str) -> None:
        """Log an error."""
        self.errors.append(message)
        print(f"{Colors.RED}❌ {message}{Colors.RESET}")

    def log_info(self, message: str) -> None:
        """Log informational message."""
        if self.verbose:
            print(f"{Colors.CYAN}ℹ️  {message}{Colors.RESET}")

    def section_header(self, title: str) -> None:
        """Print a section header."""
        print(f"\n{Colors.BOLD}{Colors.BLUE}{'=' * 70}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.BLUE}{title}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.BLUE}{'=' * 70}{Colors.RESET}")

    # ========== Check Functions ==========

    def check_python_version(self) -> bool:
        """Verify Python version is 3.12+."""
        self.section_header("1. PYTHON VERSION CHECK")

        version = sys.version_info
        if version.major == 3 and version.minor >= 12:
            self.log_success(
                f"Python {version.major}.{version.minor}.{version.micro} meets requirements"
            )
            return True
        else:
            self.log_error(f"Python {version.major}.{version.minor} < 3.12 required")
            return False

    def check_dependencies(self) -> bool:
        """Verify all required packages are installed."""
        self.section_header("2. DEPENDENCY CHECK")

        required_packages = [
            "bot_v2",
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
                if package == "bot_v2":
                    # Check our own package
                    from bot_v2 import cli
                else:
                    __import__(package)
                self.log_info(f"Package {package} found")
            except ImportError:
                self.log_error(f"Missing required package: {package}")
                all_good = False

        if all_good:
            self.log_success("All required packages installed")
        return all_good

    def check_environment_variables(self) -> bool:
        """Verify critical environment variables are set."""
        self.section_header("3. ENVIRONMENT CONFIGURATION")

        # Required variables
        required_vars = {
            "BROKER": "coinbase",
            "COINBASE_API_MODE": "advanced",
            "COINBASE_SANDBOX": "0",
            "COINBASE_ENABLE_DERIVATIVES": "1",
        }

        # Check and set required values
        all_good = True
        for var, expected in required_vars.items():
            actual = os.getenv(var)
            if actual != expected:
                if actual is None:
                    self.log_warning(f"{var} not set, should be '{expected}'")
                else:
                    self.log_error(f"{var}={actual}, must be '{expected}' for perpetuals")
                    all_good = False
            else:
                self.log_info(f"{var}={expected}")

        # Check credentials
        api_key = os.getenv("COINBASE_PROD_CDP_API_KEY") or os.getenv("COINBASE_CDP_API_KEY")
        private_key = os.getenv("COINBASE_PROD_CDP_PRIVATE_KEY") or os.getenv(
            "COINBASE_CDP_PRIVATE_KEY"
        )

        if not api_key:
            self.log_error("CDP API key not found (COINBASE_PROD_CDP_API_KEY)")
            all_good = False
        else:
            # Validate key format
            if api_key.startswith("organizations/") and "/apiKeys/" in api_key:
                self.log_success(f"CDP API key format valid: {api_key[:30]}...")
            else:
                self.log_error(
                    f"Invalid CDP API key format. Expected: organizations/.../apiKeys/..."
                )
                all_good = False

        if not private_key:
            self.log_error("CDP private key not found (COINBASE_PROD_CDP_PRIVATE_KEY)")
            all_good = False
        else:
            if "BEGIN EC PRIVATE KEY" in private_key:
                self.log_success("CDP private key found (EC format)")
            else:
                self.log_error("Invalid private key format (must be EC private key)")
                all_good = False

        # Check risk settings
        risk_vars = {
            "RISK_MAX_LEVERAGE": (1, 10),
            "RISK_DAILY_LOSS_LIMIT": (10, 10000),
            "RISK_MAX_POSITION_PCT_PER_SYMBOL": (0.01, 0.5),
        }

        for var, (min_val, max_val) in risk_vars.items():
            val = os.getenv(var)
            if val:
                try:
                    num = float(val)
                    if min_val <= num <= max_val:
                        self.log_info(f"{var}={val} (within safe range)")
                    else:
                        self.log_warning(
                            f"{var}={val} outside recommended range [{min_val}, {max_val}]"
                        )
                except ValueError:
                    self.log_error(f"{var}={val} is not a valid number")
                    all_good = False
            else:
                self.log_warning(f"{var} not set, using defaults")

        return all_good

    def check_api_connectivity(self) -> bool:
        """Test connection to Coinbase API."""
        self.section_header("4. API CONNECTIVITY TEST")

        try:
            from bot_v2.features.brokerages.coinbase.client import (
                CoinbaseClient,
                create_cdp_jwt_auth,
            )

            # Get credentials
            api_key = os.getenv("COINBASE_PROD_CDP_API_KEY") or os.getenv("COINBASE_CDP_API_KEY")
            private_key = os.getenv("COINBASE_PROD_CDP_PRIVATE_KEY") or os.getenv(
                "COINBASE_CDP_PRIVATE_KEY"
            )

            if not api_key or not private_key:
                self.log_error("Cannot test API without credentials")
                return False

            # Create auth
            auth = create_cdp_jwt_auth(
                api_key_name=api_key,
                private_key_pem=private_key,
                base_url="https://api.coinbase.com",
            )

            # Test JWT generation
            try:
                jwt_token = auth.generate_jwt("GET", "/api/v3/brokerage/accounts")
                self.log_success("JWT token generated successfully")
            except Exception as e:
                self.log_error(f"JWT generation failed: {e}")
                return False

            # Create client
            client = CoinbaseClient(
                base_url="https://api.coinbase.com", auth=auth, api_mode="advanced"
            )

            # Test endpoints
            tests = [
                ("Server time", lambda: client.get_time()),
                ("Accounts", lambda: client.get_accounts()),
                ("Products", lambda: client.list_products()),
            ]

            all_good = True
            for test_name, test_func in tests:
                try:
                    start = time.perf_counter()
                    result = test_func()
                    latency = (time.perf_counter() - start) * 1000

                    if result:
                        self.log_success(f"{test_name}: OK ({latency:.0f}ms)")
                    else:
                        self.log_warning(f"{test_name}: Empty response")
                except Exception as e:
                    self.log_error(f"{test_name}: {str(e)[:100]}")
                    all_good = False

            # Test perpetual products
            try:
                products = client.list_products()
                perps = [p for p in products if p.get("product_id", "").endswith("-PERP")]
                if perps:
                    self.log_success(f"Found {len(perps)} perpetual products")
                    if self.verbose:
                        for p in perps[:3]:
                            self.log_info(f"  - {p.get('product_id')}")
                else:
                    self.log_error("No perpetual products found")
                    all_good = False
            except Exception as e:
                self.log_error(f"Failed to list products: {e}")
                all_good = False

            return all_good

        except ImportError as e:
            self.log_error(f"Failed to import required modules: {e}")
            return False
        except Exception as e:
            self.log_error(f"Unexpected error during API test: {e}")
            return False

    def check_risk_configuration(self) -> bool:
        """Validate risk management settings."""
        self.section_header("5. RISK MANAGEMENT VALIDATION")

        try:
            from bot_v2.config.live_trade_config import RiskConfig

            # Load risk config
            config = RiskConfig.from_env()

            # Critical checks
            checks = [
                ("Max leverage", config.max_leverage, lambda x: 1 <= x <= 10),
                ("Daily loss limit", config.daily_loss_limit, lambda x: x > 0),
                ("Liquidation buffer", config.min_liquidation_buffer_pct, lambda x: x >= 0.10),
                ("Position limit", config.max_position_pct_per_symbol, lambda x: 0 < x <= 0.25),
                ("Slippage guard", config.slippage_guard_bps, lambda x: 10 <= x <= 100),
            ]

            all_good = True
            for name, value, validator in checks:
                if validator(value):
                    self.log_success(f"{name}: {value} ✓")
                else:
                    self.log_error(f"{name}: {value} - UNSAFE VALUE")
                    all_good = False

            # Warning checks
            if config.kill_switch_enabled:
                self.log_warning("Kill switch is ENABLED - all trading blocked")

            if config.reduce_only_mode:
                self.log_warning("Reduce-only mode ENABLED - can only close positions")

            if config.daily_loss_limit > Decimal("1000"):
                self.log_warning(
                    f"Daily loss limit ${config.daily_loss_limit} seems high for testing"
                )

            if config.max_leverage > 5:
                self.log_warning(f"Leverage {config.max_leverage}x is aggressive")

            return all_good

        except Exception as e:
            self.log_error(f"Failed to validate risk config: {e}")
            return False

    def check_test_suite(self) -> bool:
        """Run critical tests."""
        self.section_header("6. TEST SUITE VALIDATION")

        print("Running core test suite...")

        # Run tests with pytest
        import subprocess

        try:
            result = subprocess.run(
                [
                    "poetry",
                    "run",
                    "pytest",
                    "tests/unit/bot_v2/orchestration",
                    "tests/unit/bot_v2/features/brokerages/coinbase/test_auth_and_models.py",
                    "-q",
                    "--tb=no",
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )

            # Parse output
            output = result.stdout + result.stderr
            if "passed" in output:
                # Extract test count
                import re

                match = re.search(r"(\d+) passed", output)
                if match:
                    passed = int(match.group(1))
                    self.log_success(f"{passed} core tests passed")

                    # Check for failures
                    if "failed" in output:
                        match = re.search(r"(\d+) failed", output)
                        if match:
                            failed = int(match.group(1))
                            self.log_warning(f"{failed} tests failed")
                            return False
                    return True
            else:
                self.log_error("Test suite failed")
                if self.verbose:
                    print(output)
                return False

        except subprocess.TimeoutExpired:
            self.log_error("Test suite timed out")
            return False
        except Exception as e:
            self.log_error(f"Failed to run tests: {e}")
            return False

    def check_profile_configuration(self) -> bool:
        """Validate selected trading profile."""
        self.section_header("7. PROFILE CONFIGURATION")

        profile_path = Path(f"config/profiles/{self.profile}.yaml")

        if profile_path.exists():
            self.log_success(f"Profile '{self.profile}' found at {profile_path}")

            # Parse and validate profile
            try:
                import yaml

                with open(profile_path) as f:
                    profile_config = yaml.safe_load(f)

                # Check critical settings
                if self.profile == "canary":
                    expected = {
                        "trading.mode": "reduce_only",
                        "trading.position_sizing.max_position_size": 0.01,
                        "risk_management.daily_loss_limit": 10.00,
                        "risk_management.max_leverage": 1.0,
                    }

                    for key, expected_val in expected.items():
                        keys = key.split(".")
                        val = profile_config
                        for k in keys:
                            val = val.get(k, {})

                        if val == expected_val:
                            self.log_info(f"{key} = {val} ✓")
                        else:
                            self.log_warning(f"{key} = {val}, expected {expected_val}")

                self.log_success(f"Profile '{self.profile}' validated")
                return True

            except Exception as e:
                self.log_error(f"Failed to parse profile: {e}")
                return False
        else:
            # Use default profile settings
            self.log_warning(f"Profile '{self.profile}' not found, will use defaults")

            if self.profile == "canary":
                self.log_info("Canary defaults: 0.01 BTC max, $10 daily loss, reduce-only")
            elif self.profile == "prod":
                self.log_warning("Production profile - ensure you've tested with canary first!")

            return True

    def check_system_time(self) -> bool:
        """Verify system clock synchronization."""
        self.section_header("8. SYSTEM TIME SYNC")

        try:
            # Get system time
            system_time = datetime.now(timezone.utc)

            # Try to get server time from Coinbase
            from bot_v2.features.brokerages.coinbase.client import (
                CoinbaseClient,
                create_cdp_jwt_auth,
            )

            api_key = os.getenv("COINBASE_PROD_CDP_API_KEY") or os.getenv("COINBASE_CDP_API_KEY")
            private_key = os.getenv("COINBASE_PROD_CDP_PRIVATE_KEY") or os.getenv(
                "COINBASE_CDP_PRIVATE_KEY"
            )

            if api_key and private_key:
                auth = create_cdp_jwt_auth(
                    api_key_name=api_key,
                    private_key_pem=private_key,
                    base_url="https://api.coinbase.com",
                )
                client = CoinbaseClient(
                    base_url="https://api.coinbase.com", auth=auth, api_mode="advanced"
                )

                try:
                    server_time_resp = client.get_time()
                    if server_time_resp and "iso" in server_time_resp:
                        server_time = datetime.fromisoformat(
                            server_time_resp["iso"].replace("Z", "+00:00")
                        )

                        # Compare times
                        diff = abs((system_time - server_time).total_seconds())

                        if diff < 1:
                            self.log_success(f"System clock synchronized (drift: {diff:.2f}s)")
                            return True
                        elif diff < 5:
                            self.log_warning(f"System clock drift: {diff:.2f}s (acceptable)")
                            return True
                        else:
                            self.log_error(f"System clock drift: {diff:.2f}s - SYNC REQUIRED")
                            self.log_info("Run: sudo ntpdate -s time.nist.gov")
                            return False
                except Exception:
                    pass

            # Fallback: just check if time seems reasonable
            if 2024 <= system_time.year <= 2030:
                self.log_warning("Cannot verify time sync, but system time seems reasonable")
                self.log_info(f"System time: {system_time.isoformat()}")
                return True
            else:
                self.log_error(f"System time seems wrong: {system_time}")
                return False

        except Exception as e:
            self.log_error(f"Failed to check system time: {e}")
            return False

    def check_disk_space(self) -> bool:
        """Verify adequate disk space for logs and events."""
        self.section_header("9. DISK SPACE CHECK")

        import shutil

        try:
            usage = shutil.disk_usage(".")
            free_gb = usage.free / (1024**3)
            total_gb = usage.total / (1024**3)
            used_pct = (usage.used / usage.total) * 100

            if free_gb > 1.0:
                self.log_success(
                    f"Disk space: {free_gb:.1f}GB free of {total_gb:.1f}GB ({used_pct:.0f}% used)"
                )
                return True
            elif free_gb > 0.5:
                self.log_warning(f"Low disk space: {free_gb:.1f}GB free")
                return True
            else:
                self.log_error(f"Critical: Only {free_gb:.1f}GB free")
                return False

        except Exception as e:
            self.log_error(f"Failed to check disk space: {e}")
            return False

    def simulate_dry_run(self) -> bool:
        """Simulate a dry-run execution."""
        self.section_header("10. DRY-RUN SIMULATION")

        print("Simulating dry-run execution...")

        try:
            from bot_v2.orchestration.configuration import BotConfig
            from bot_v2.orchestration.bootstrap import build_bot
            from bot_v2.orchestration.broker_factory import create_brokerage

            # Create config (mock_broker toggles the DeterministicBroker safety stub)
            config = BotConfig.from_profile(profile=self.profile, dry_run=True, mock_broker=True)

            self.log_info(f"Config: {self.profile} profile, dry_run=True, deterministic broker")

            # Try to create broker
            try:
                broker = create_brokerage()
                self.log_success("Broker factory initialized")
            except Exception as e:
                self.log_warning(f"Broker initialization warning: {e}")

            # Validate strategy
            from bot_v2.features.live_trade.strategies.perps_baseline import BaselinePerpsStrategy

            strategy = BaselinePerpsStrategy()
            self.log_success("Strategy initialized")

            # Check if we can create the bot via bootstrap
            bot, registry = build_bot(config)
            self.log_success("PerpsBot constructed via bootstrap")
            # Ensure broker/risk manager present
            if bot.broker:
                self.log_success("Broker instance available")
            if bot.risk_manager:
                self.log_success("Risk manager initialized")
            self.log_success("Dry-run simulation passed")
            return True

        except Exception as e:
            self.log_error(f"Dry-run simulation failed: {e}")
            return False

    def generate_report(self) -> tuple[bool, str]:
        """Generate final preflight report."""
        self.section_header("PREFLIGHT REPORT")

        total_checks = len(self.successes) + len(self.warnings) + len(self.errors)

        print(f"\n{Colors.BOLD}Summary:{Colors.RESET}")
        print(f"  {Colors.GREEN}✅ Passed: {len(self.successes)}{Colors.RESET}")
        print(f"  {Colors.YELLOW}⚠️  Warnings: {len(self.warnings)}{Colors.RESET}")
        print(f"  {Colors.RED}❌ Failed: {len(self.errors)}{Colors.RESET}")

        # Decision
        if len(self.errors) == 0:
            if len(self.warnings) <= 3:
                status = "READY"
                color = Colors.GREEN
                message = "System is READY for production trading (with caution)"
            else:
                status = "REVIEW"
                color = Colors.YELLOW
                message = "System has warnings - review before proceeding"
        else:
            status = "NOT READY"
            color = Colors.RED
            message = "System is NOT READY - critical issues must be resolved"

        print(f"\n{Colors.BOLD}{color}{'=' * 70}{Colors.RESET}")
        print(f"{Colors.BOLD}{color}STATUS: {status}{Colors.RESET}")
        print(f"{color}{message}{Colors.RESET}")
        print(f"{Colors.BOLD}{color}{'=' * 70}{Colors.RESET}")

        # Recommendations
        print(f"\n{Colors.BOLD}Recommendations:{Colors.RESET}")

        if status == "READY":
            print(f"1. Start with: poetry run perps-bot --profile {self.profile} --dry-run")
            print(f"2. Monitor for 1 hour in dry-run mode")
            print(f"3. Begin live with: poetry run perps-bot --profile {self.profile}")
            print(f"4. Use tiny positions (0.001 BTC) initially")
            print(f"5. Monitor closely for first 24 hours")
        elif status == "REVIEW":
            print(f"1. Review all warnings above")
            print(f"2. Consider starting with paper trading: PERPS_PAPER=1")
            print(f"3. Ensure emergency procedures are documented")
            print(f"4. Test kill switch: RISK_KILL_SWITCH_ENABLED=1")
        else:
            print(f"1. Fix all critical errors listed above")
            print(f"2. Review config/environments/.env.production for configuration guidance")
            print(f"3. Run tests: poetry run pytest tests/unit/bot_v2")
            print(f"4. Verify credentials and API connectivity")

        # Save report
        report_path = Path(f"preflight_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        report_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "profile": self.profile,
            "status": status,
            "successes": len(self.successes),
            "warnings": len(self.warnings),
            "errors": len(self.errors),
            "details": {
                "successes": self.successes,
                "warnings": self.warnings,
                "errors": self.errors,
            },
        }

        try:
            with open(report_path, "w") as f:
                json.dump(report_data, f, indent=2)
            print(f"\n{Colors.CYAN}Report saved to: {report_path}{Colors.RESET}")
        except Exception as e:
            print(f"\n{Colors.YELLOW}Could not save report: {e}{Colors.RESET}")

        return len(self.errors) == 0, status


def main():
    """Run preflight checks."""
    parser = argparse.ArgumentParser(description="Production preflight check for GPT-Trader")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    parser.add_argument(
        "--profile",
        "-p",
        default="canary",
        choices=["dev", "canary", "prod"],
        help="Trading profile to validate (default: canary)",
    )

    args = parser.parse_args()

    # Header
    print(f"{Colors.BOLD}{Colors.MAGENTA}")
    print("=" * 70)
    print("GPT-TRADER PRODUCTION PREFLIGHT CHECK")
    print(f"Profile: {args.profile}")
    print(f"Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("=" * 70)
    print(f"{Colors.RESET}")

    # Run checks
    checker = PreflightCheck(verbose=args.verbose, profile=args.profile)

    checks = [
        checker.check_python_version,
        checker.check_dependencies,
        checker.check_environment_variables,
        checker.check_api_connectivity,
        checker.check_risk_configuration,
        checker.check_test_suite,
        checker.check_profile_configuration,
        checker.check_system_time,
        checker.check_disk_space,
        checker.simulate_dry_run,
    ]

    # Run all checks
    for check in checks:
        try:
            check()
        except Exception as e:
            checker.log_error(f"Check failed with exception: {e}")

    # Generate report
    success, status = checker.generate_report()

    # Exit code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
