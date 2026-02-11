"""Remediation hints for common preflight checks."""

from __future__ import annotations

from typing import Final

DEFAULT_REMEDIATION_HINT: Final[str] = (
    "Review the preflight failure and consult the documentation for the failing check."
)

_REMEDIATION_HINTS: dict[str, str] = {
    "check_python_version": (
        "Install Python 3.12 with `uv python install 3.12` and activate the new interpreter."
    ),
    "check_dependencies": (
        "Run `uv sync --all-extras --dev` to install required packages (including live-trade extras)."
    ),
    "check_environment_variables": (
        "Fill in missing variables from `config/environments/.env.template` or the profile defaults."
    ),
    "check_api_connectivity": (
        "Verify Coinbase API credentials (export API key + private key) and ensure network access."
    ),
    "check_key_permissions": (
        "Ensure your API key has portfolio view + trade permissions in the Coinbase dashboard."
    ),
    "check_risk_configuration": (
        "Correct the risk configuration files or overrides before resuming preflight."
    ),
    "check_pretrade_diagnostics": (
        "Resolve diagnostics failures (timeouts, missing accounts) before trading."
    ),
    "check_test_suite": (
        "Run `uv run pytest tests/unit/gpt_trader` to inspect failing tests and dependencies."
    ),
    "check_profile_configuration": (
        "Validate the profile YAML (`config/profiles/<profile>.yaml`) for syntax or missing fields."
    ),
    "check_system_time": (
        "Sync the host clock (e.g., `sudo ntpdate -u time.google.com` or enable NTP via timedatectl)."
    ),
    "check_disk_space": (
        "Free disk space or move `GPT_TRADER_DATA_DIR`/`runtime_data` onto a larger volume."
    ),
    "simulate_dry_run": (
        "Inspect the dry-run failure, fix any configuration issues, and rerun with `--verbose`."
    ),
    "check_event_store_redaction": (
        "Fix redaction findings by removing secrets from runtime events before rerunning."
    ),
    "check_readiness_report": (
        "Refresh the readiness report (e.g., `make preflight-readiness`) or ensure report data exists."
    ),
}


def get_remediation_hint(check_name: str | None) -> str:
    """Return a remediation hint for the provided check name."""
    if not check_name:
        return DEFAULT_REMEDIATION_HINT
    return _REMEDIATION_HINTS.get(check_name, DEFAULT_REMEDIATION_HINT)
