#!/usr/bin/env python3
"""
Config Doctor - Validate GPT-Trader configuration files

Usage:
  python scripts/tools/config_doctor.py --check all
  python scripts/tools/config_doctor.py --check env
  python scripts/tools/config_doctor.py --check risk
  python scripts/tools/config_doctor.py --compare .env config/environments/.env.template
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, ValidationError, field_validator


# =============================================================================
# Pydantic Validation Schemas
# =============================================================================


class RiskConfig(BaseModel):
    """Schema for risk configuration files."""

    max_leverage: int = Field(ge=1, le=20, description="Maximum leverage allowed")
    daily_loss_limit: str | int = Field(description="Daily loss limit in USD")
    max_exposure_pct: float = Field(ge=0, le=1, description="Maximum portfolio exposure")
    max_position_pct_per_symbol: float = Field(
        ge=0, le=1, description="Maximum position size per symbol"
    )
    slippage_guard_bps: int = Field(ge=0, le=500, description="Slippage guard in basis points")

    # Optional fields
    min_liquidation_buffer_pct: float | None = Field(
        default=None, ge=0, le=1, description="Minimum liquidation buffer"
    )
    enable_pre_trade_liq_projection: bool | None = None
    kill_switch_enabled: bool | None = None
    reduce_only_mode: bool | None = None
    max_mark_staleness_seconds: int | None = Field(default=None, ge=0)
    enable_volatility_circuit_breaker: bool | None = None
    enable_market_impact_guard: bool | None = None
    max_market_impact_bps: str | int | None = None
    enable_dynamic_position_sizing: bool | None = None
    position_sizing_method: str | None = None
    position_sizing_multiplier: float | None = Field(default=None, ge=0)

    # Per-symbol settings (optional)
    leverage_max_per_symbol: dict[str, int] | None = None
    max_notional_per_symbol: dict[str, str | int] | None = None

    # Day/night leverage (optional)
    daytime_start_utc: str | None = None
    daytime_end_utc: str | None = None
    day_leverage_max_per_symbol: dict[str, int] | None = None
    night_leverage_max_per_symbol: dict[str, int] | None = None

    @field_validator("daily_loss_limit")
    @classmethod
    def validate_daily_loss_limit(cls, v: str | int) -> str | int:
        """Validate daily loss limit is positive."""
        if isinstance(v, str):
            try:
                float_val = float(v)
                if float_val <= 0:
                    raise ValueError("daily_loss_limit must be positive")
            except ValueError as e:
                if "could not convert" in str(e):
                    raise ValueError(f"daily_loss_limit must be numeric, got: {v}") from e
                raise
        elif isinstance(v, int) and v <= 0:
            raise ValueError("daily_loss_limit must be positive")
        return v


class ProfileConfig(BaseModel):
    """Schema for profile configuration files (basic validation)."""

    # Profile configs can be very diverse, so we do minimal validation
    # Just ensure basic structure is present
    pass  # Extend as needed based on actual profile structure


# =============================================================================
# Environment File Parsing
# =============================================================================


def parse_env_file(env_path: Path) -> dict[str, str]:
    """Parse .env file into key-value dict."""
    env_vars = {}

    if not env_path.exists():
        return env_vars

    with open(env_path) as f:
        for line in f:
            line = line.strip()
            # Skip comments and empty lines
            if not line or line.startswith("#"):
                continue

            # Parse KEY=VALUE
            if "=" in line:
                key, value = line.split("=", 1)
                env_vars[key.strip()] = value.strip()

    return env_vars


# =============================================================================
# Validation Functions
# =============================================================================


def check_env_file(env_path: Path, template_path: Path) -> list[str]:
    """Compare actual .env against template, flag missing/extra vars."""
    issues = []

    # Load both files
    env_vars = parse_env_file(env_path)
    template_vars = parse_env_file(template_path)

    # Extract required variables (non-empty in template)
    required_vars = {k for k, v in template_vars.items() if v and v != ""}

    # Find missing required vars
    missing = required_vars - set(env_vars.keys())
    if missing:
        issues.append(f"Missing required variables: {', '.join(sorted(missing))}")

    # Find extra unexpected vars (not necessarily bad, but worth noting)
    extra = set(env_vars.keys()) - set(template_vars.keys())
    if extra:
        issues.append(f"Extra variables (not in template): {', '.join(sorted(extra))}")

    # Check for empty values in required fields
    empty_required = [k for k in required_vars if k in env_vars and not env_vars[k]]
    if empty_required:
        issues.append(f"Required variables are empty: {', '.join(sorted(empty_required))}")

    return issues


def validate_risk_config(risk_path: Path) -> list[str]:
    """Validate risk config against Pydantic schema."""
    issues = []

    try:
        # Load config based on file type
        if risk_path.suffix in [".yaml", ".yml"]:
            with open(risk_path) as f:
                config_data = yaml.safe_load(f)
        elif risk_path.suffix == ".json":
            with open(risk_path) as f:
                config_data = json.load(f)
        else:
            return [f"Unknown config format: {risk_path.suffix} (expected .yaml/.yml/.json)"]

        # Validate against schema
        try:
            RiskConfig(**config_data)
        except ValidationError as e:
            for error in e.errors():
                field = ".".join(str(loc) for loc in error["loc"])
                msg = error["msg"]
                issues.append(f"{field}: {msg}")

    except yaml.YAMLError as e:
        issues.append(f"YAML syntax error: {e}")
    except json.JSONDecodeError as e:
        issues.append(f"JSON syntax error: {e}")
    except Exception as e:
        issues.append(f"Unexpected error loading config: {e}")

    return issues


def check_duplicate_configs(config_dir: Path) -> list[str]:
    """Check for duplicate config files (same name, different format)."""
    issues = []

    # Find all config files
    yaml_files = {f.stem for f in config_dir.glob("*.yaml")} | {
        f.stem for f in config_dir.glob("*.yml")
    }
    json_files = {f.stem for f in config_dir.glob("*.json")}

    # Find duplicates
    duplicates = yaml_files & json_files
    if duplicates:
        for dup in sorted(duplicates):
            issues.append(
                f"Duplicate config '{dup}' exists in both YAML and JSON formats. "
                "Consolidate to a single format (YAML recommended)."
            )

    return issues


# =============================================================================
# Main CLI
# =============================================================================


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Validate GPT-Trader configuration files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check all configurations
  python scripts/tools/config_doctor.py --check all

  # Check only environment files
  python scripts/tools/config_doctor.py --check env

  # Check only risk configurations
  python scripts/tools/config_doctor.py --check risk

  # Compare .env against template
  python scripts/tools/config_doctor.py --compare .env config/environments/.env.template
        """,
    )

    parser.add_argument(
        "--check",
        choices=["all", "env", "risk", "profiles"],
        default="all",
        help="Type of configuration to check",
    )

    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("ENV_FILE", "TEMPLATE_FILE"),
        help="Compare an .env file against a template",
    )

    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with error code if any issues found (for CI)",
    )

    args = parser.parse_args()

    issues_found = False
    total_issues = 0

    print("=" * 80)
    print("GPT-TRADER CONFIG DOCTOR")
    print("=" * 80)
    print()

    # Compare mode
    if args.compare:
        env_path, template_path = Path(args.compare[0]), Path(args.compare[1])

        print(f"Comparing {env_path} against {template_path}...")
        print()

        if not env_path.exists():
            print(f"❌ Environment file not found: {env_path}")
            return 1

        if not template_path.exists():
            print(f"❌ Template file not found: {template_path}")
            return 1

        issues = check_env_file(env_path, template_path)
        if issues:
            print(f"⚠️  Issues found in {env_path}:")
            for issue in issues:
                print(f"   - {issue}")
            total_issues += len(issues)
            issues_found = True
        else:
            print(f"✅ {env_path} matches template requirements")

    # Check mode
    else:
        # Check environment files
        if args.check in ["all", "env"]:
            print("Checking environment configurations...")
            print()

            env_dir = Path("config/environments")
            template_path = env_dir / ".env.template"

            if template_path.exists():
                # Check if .env exists in root
                root_env = Path(".env")
                if root_env.exists():
                    issues = check_env_file(root_env, template_path)
                    if issues:
                        print(f"⚠️  Issues in .env:")
                        for issue in issues:
                            print(f"   - {issue}")
                        total_issues += len(issues)
                        issues_found = True
                    else:
                        print("✅ .env is valid")
                else:
                    print("ℹ️  No .env file found in root (expected - create from template)")

                print()
            else:
                print(f"⚠️  Template not found: {template_path}")
                issues_found = True

        # Check risk configurations
        if args.check in ["all", "risk"]:
            print("Checking risk configurations...")
            print()

            risk_dir = Path("config/risk")
            if risk_dir.exists():
                # Check for duplicates first
                dup_issues = check_duplicate_configs(risk_dir)
                if dup_issues:
                    print("⚠️  Duplicate configuration files:")
                    for issue in dup_issues:
                        print(f"   - {issue}")
                    total_issues += len(dup_issues)
                    issues_found = True
                    print()

                # Validate each config file
                for config_file in (
                    sorted(risk_dir.glob("*.yaml"))
                    + sorted(risk_dir.glob("*.yml"))
                    + sorted(risk_dir.glob("*.json"))
                ):
                    if config_file.name == "README.md":
                        continue

                    print(f"Validating {config_file.name}...")
                    issues = validate_risk_config(config_file)

                    if issues:
                        print(f"⚠️  Issues in {config_file.name}:")
                        for issue in issues:
                            print(f"   - {issue}")
                        total_issues += len(issues)
                        issues_found = True
                    else:
                        print(f"✅ {config_file.name} is valid")

                    print()
            else:
                print(f"⚠️  Risk config directory not found: {risk_dir}")
                issues_found = True

        # Check profiles (basic check)
        if args.check in ["all", "profiles"]:
            print("Checking profile configurations...")
            print()

            profile_dir = Path("config/profiles")
            if profile_dir.exists():
                profile_files = list(profile_dir.glob("*.yaml")) + list(profile_dir.glob("*.yml"))

                if not profile_files:
                    print("⚠️  No profile configurations found")
                    issues_found = True
                else:
                    for profile_file in sorted(profile_files):
                        print(f"Found profile: {profile_file.name}")

                        # Basic YAML syntax check
                        try:
                            with open(profile_file) as f:
                                yaml.safe_load(f)
                            print(f"✅ {profile_file.name} has valid YAML syntax")
                        except yaml.YAMLError as e:
                            print(f"❌ {profile_file.name} has YAML syntax error: {e}")
                            total_issues += 1
                            issues_found = True

                        print()
            else:
                print(f"⚠️  Profile directory not found: {profile_dir}")
                issues_found = True

    # Summary
    print("=" * 80)
    if not issues_found:
        print("✅ All configurations are valid!")
        print("=" * 80)
        return 0
    else:
        print(f"⚠️  Found {total_issues} configuration issue(s)")
        print("=" * 80)
        print()
        print("Please fix the issues above before deploying.")

        if args.strict:
            return 1

        return 0


if __name__ == "__main__":
    sys.exit(main())
