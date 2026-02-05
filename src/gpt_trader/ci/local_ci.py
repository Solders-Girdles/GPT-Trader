from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

ORCHESTRATION_GUARD_SCRIPT = "\n".join(
    [
        "# The gpt_trader.orchestration package was removed in v3.0",
        "# Fail if ANY file imports gpt_trader.orchestration",
        'if grep -rn -E "(from|import)\\s+gpt_trader\\.orchestration" src tests scripts --include="*.py"; then',
        '  echo "::error::gpt_trader.orchestration was removed in v3.0"',
        '  echo "Use canonical paths: app.*, features.live_trade.*, features.brokerages.*"',
        '  echo "See docs/DEPRECATIONS.md for migration guidance."',
        "  exit 1",
        "fi",
        'echo "No orchestration imports found - package was removed in v3.0."',
    ]
)

AGENT_HEALTH_SCRIPT = "\n".join(
    [
        "set -u",
        "set -o pipefail",
        "mkdir -p var/agents/health",
        "set +e",
        "make agent-health-fast AGENT_HEALTH_FAST_QUALITY_CHECKS=none",
        "status=$?",
        "set -e",
        "if [ -f var/agents/health/health_report.json ]; then",
        "  uv run python - <<'PY' | tee var/agents/health/health_report.txt",
        "from scripts.agents.health_report import format_text_report",
        "import json",
        'with open("var/agents/health/health_report.json") as handle:',
        "    report = json.load(handle)",
        "print(format_text_report(report))",
        "PY",
        "fi",
        "exit $status",
    ]
)


@dataclass(frozen=True)
class PlannedStep:
    label: str
    command: Sequence[str]
    env: dict[str, str] | None = None
    enabled: bool = True
    skip_reason: str | None = None


@dataclass(frozen=True)
class StepResult:
    label: str
    status: str
    return_code: int | None
    note: str | None = None


def find_repo_root(start_path: Path) -> Path:
    for candidate in [start_path, *start_path.parents]:
        if (candidate / "pyproject.toml").is_file():
            return candidate
    message = f"Could not find repo root from {start_path}"
    raise RuntimeError(message)


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run local CI checks that mirror required GitHub Actions jobs."
    )
    parser.add_argument(
        "--include-snapshots",
        action="store_true",
        help="Include TUI snapshot tests (CI: tui-snapshots).",
    )
    parser.add_argument(
        "--include-property-tests",
        action="store_true",
        help="Include property tests (CI: property-tests).",
    )
    parser.add_argument(
        "--include-contract-tests",
        action="store_true",
        help="Include contract tests (CI: contract-tests).",
    )
    parser.add_argument(
        "--include-agent-health",
        action="store_true",
        help="Include agent-health fast checks (CI: agent-health).",
    )
    return parser.parse_args(argv)


def build_steps(args: argparse.Namespace) -> list[PlannedStep]:
    test_env = {
        "GPT_TRADER_STRICT_CONTAINER": "1",
        "PYTHONWARNINGS": "default",
    }

    steps = [
        PlannedStep(
            label="Lint (Ruff)",
            command=["uv", "run", "ruff", "check", "."],
        ),
        PlannedStep(
            label="Lint (Ruff - critical scripts)",
            command=[
                "uv",
                "run",
                "ruff",
                "check",
                "scripts/ops",
                "scripts/backtest_runner.py",
                "scripts/perps_dashboard.py",
                "scripts/monitoring/export_metrics.py",
                "scripts/monitoring/canary_reduce_only_test.py",
                "scripts/monitoring/manage_logs.py",
                "scripts/production_preflight.py",
                "scripts/readiness_window.py",
                "scripts/test_api_connectivity.py",
                "scripts/test_paper_broker.py",
            ],
        ),
        PlannedStep(
            label="Format (Black)",
            command=["uv", "run", "black", "--check", "."],
        ),
        PlannedStep(
            label="Guard against orchestration imports (removed in v3.0)",
            command=["bash", "-lc", ORCHESTRATION_GUARD_SCRIPT],
        ),
        PlannedStep(
            label="Guard deprecation registry",
            command=["python", "scripts/ci/check_deprecation_registry.py"],
        ),
        PlannedStep(
            label="Docs link audit",
            command=["python", "scripts/maintenance/docs_link_audit.py"],
        ),
        PlannedStep(
            label="Docs reachability check",
            command=["python", "scripts/maintenance/docs_reachability_check.py"],
        ),
        PlannedStep(
            label="Type Check (MyPy)",
            command=["uv", "run", "mypy", "src"],
        ),
        PlannedStep(
            label="Agent health (fast, dev profile)",
            command=["bash", "-lc", AGENT_HEALTH_SCRIPT],
            enabled=args.include_agent_health,
            skip_reason="use --include-agent-health",
        ),
        PlannedStep(
            label="Agent artifacts freshness",
            command=["uv", "run", "agent-regenerate", "--verify"],
        ),
        PlannedStep(
            label="TUI CSS check",
            command=["python", "scripts/ci/check_tui_css_up_to_date.py"],
        ),
        PlannedStep(
            label="Check test hygiene",
            command=["uv", "run", "python", "scripts/ci/check_test_hygiene.py"],
        ),
        PlannedStep(
            label="Check legacy patterns",
            command=["uv", "run", "python", "scripts/ci/check_legacy_patterns.py"],
        ),
        PlannedStep(
            label="Check import boundaries",
            command=["uv", "run", "python", "scripts/ci/check_import_boundaries.py"],
        ),
        PlannedStep(
            label="Readiness gate (3-day streak)",
            command=[
                "python",
                "scripts/ci/check_readiness_gate.py",
                "--profile",
                "canary",
            ],
        ),
        PlannedStep(
            label="Check legacy triage alignment",
            command=["uv", "run", "python", "scripts/ci/check_legacy_test_triage.py"],
        ),
        PlannedStep(
            label="Check dedupe manifest",
            command=["uv", "run", "python", "scripts/ci/check_dedupe_manifest.py", "--strict"],
        ),
        PlannedStep(
            label="Check triage backlog",
            command=["make", "test-triage-check"],
        ),
        PlannedStep(
            label="Unit tests (core)",
            command=[
                "uv",
                "run",
                "pytest",
                "tests/unit",
                "-n",
                "auto",
                "-q",
                "--ignore-glob=tests/unit/gpt_trader/tui/test_snapshots_*.py",
            ],
            env=test_env,
        ),
        PlannedStep(
            label="TUI snapshot tests",
            command=[
                "uv",
                "run",
                "pytest",
                "tests/unit/gpt_trader/tui/test_snapshots_*.py",
                "-v",
            ],
            env=test_env,
            enabled=args.include_snapshots,
            skip_reason="use --include-snapshots",
        ),
        PlannedStep(
            label="Property tests",
            command=["uv", "run", "pytest", "tests/property", "-v"],
            env=test_env,
            enabled=args.include_property_tests,
            skip_reason="use --include-property-tests",
        ),
        PlannedStep(
            label="Contract tests",
            command=["uv", "run", "pytest", "tests/contract", "-v"],
            env=test_env,
            enabled=args.include_contract_tests,
            skip_reason="use --include-contract-tests",
        ),
    ]
    return steps


def format_command(command: Sequence[str]) -> str:
    return shlex.join(command)


def run_steps(steps: Sequence[PlannedStep], repo_root: Path) -> list[StepResult]:
    results: list[StepResult] = []
    total_enabled = sum(1 for step in steps if step.enabled)
    run_index = 0
    for step in steps:
        if not step.enabled:
            results.append(
                StepResult(label=step.label, status="skip", return_code=None, note=step.skip_reason)
            )
            continue
        run_index += 1
        print(f"\n==> [{run_index}/{total_enabled}] {step.label}", flush=True)
        print(f"$ {format_command(step.command)}", flush=True)
        command_env = os.environ.copy()
        if step.env:
            command_env.update(step.env)
        try:
            completed = subprocess.run(
                step.command,
                check=False,
                cwd=repo_root,
                env=command_env,
            )
        except FileNotFoundError as exc:
            print(f"Command failed to start: {exc}", file=sys.stderr)
            results.append(
                StepResult(
                    label=step.label,
                    status="fail",
                    return_code=127,
                    note="command not found",
                )
            )
            continue
        status = "pass" if completed.returncode == 0 else "fail"
        results.append(
            StepResult(label=step.label, status=status, return_code=completed.returncode)
        )
    return results


def print_summary(results: Sequence[StepResult]) -> None:
    print("\nSummary:", flush=True)
    for result in results:
        status = result.status.upper().ljust(5)
        line = f"{status} {result.label}"
        if result.status == "fail" and result.return_code is not None:
            line = f"{line} (exit {result.return_code})"
        if result.status == "skip" and result.note:
            line = f"{line} ({result.note})"
        print(line, flush=True)

    failed = [result for result in results if result.status == "fail"]
    if failed:
        print(f"\nLocal CI failed: {len(failed)} check(s) failed.", flush=True)
    else:
        print("\nLocal CI passed.", flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = find_repo_root(Path(__file__).resolve())
    os.chdir(repo_root)
    steps = build_steps(args)
    results = run_steps(steps, repo_root)
    print_summary(results)
    return 1 if any(result.status == "fail" for result in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
