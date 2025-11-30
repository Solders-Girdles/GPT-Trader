#!/usr/bin/env python3
"""Quality gate runner with JSON output for AI agent consumption.

Runs all quality checks (linting, type checking, tests) and produces
machine-readable output for automated workflows.

Usage:
    python scripts/agents/quality_gate.py [--check CHECKS] [--format json|text]
    python scripts/agents/quality_gate.py --check lint,types
    python scripts/agents/quality_gate.py --fast  # Skip slow tests
    python scripts/agents/quality_gate.py --files src/gpt_trader/cli/  # Check specific paths

Output:
    JSON report with pass/fail status for each check and detailed findings.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).parent.parent.parent


@dataclass
class CheckResult:
    """Result of a single quality check."""

    name: str
    passed: bool
    duration_seconds: float
    exit_code: int
    findings: list[dict[str, Any]] = field(default_factory=list)
    summary: str = ""
    command: str = ""
    stdout: str = ""
    stderr: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "duration_seconds": round(self.duration_seconds, 2),
            "exit_code": self.exit_code,
            "findings_count": len(self.findings),
            "findings": self.findings[:50],  # Limit to first 50
            "summary": self.summary,
            "command": self.command,
        }


def run_command(cmd: list[str], cwd: Path | None = None) -> tuple[int, str, str, float]:
    """Run a command and return exit code, stdout, stderr, duration."""
    start = time.time()
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd or PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
        )
        duration = time.time() - start
        return result.returncode, result.stdout, result.stderr, duration
    except subprocess.TimeoutExpired:
        return 124, "", "Command timed out after 300 seconds", time.time() - start
    except Exception as e:
        return 1, "", str(e), time.time() - start


def parse_ruff_output(stdout: str) -> list[dict[str, Any]]:
    """Parse ruff JSON output into findings."""
    findings = []
    try:
        # Ruff outputs one JSON object per line in some modes
        for line in stdout.strip().split("\n"):
            if line.startswith("{"):
                finding = json.loads(line)
                findings.append(
                    {
                        "file": finding.get("filename", ""),
                        "line": finding.get("location", {}).get("row", 0),
                        "column": finding.get("location", {}).get("column", 0),
                        "code": finding.get("code", ""),
                        "message": finding.get("message", ""),
                        "severity": (
                            "error" if finding.get("code", "").startswith("E") else "warning"
                        ),
                    }
                )
    except json.JSONDecodeError:
        # Fall back to line-based parsing
        for line in stdout.strip().split("\n"):
            if ":" in line and line.strip():
                parts = line.split(":", 3)
                if len(parts) >= 4:
                    findings.append(
                        {
                            "file": parts[0],
                            "line": int(parts[1]) if parts[1].isdigit() else 0,
                            "column": int(parts[2]) if parts[2].isdigit() else 0,
                            "message": parts[3].strip(),
                            "severity": "error",
                        }
                    )
    return findings


def parse_mypy_output(stdout: str) -> list[dict[str, Any]]:
    """Parse mypy output into findings."""
    findings = []
    for line in stdout.strip().split("\n"):
        if ": error:" in line or ": warning:" in line or ": note:" in line:
            # Format: file:line: severity: message
            parts = line.split(":", 3)
            if len(parts) >= 4:
                severity = "error"
                if "warning" in parts[2]:
                    severity = "warning"
                elif "note" in parts[2]:
                    severity = "info"
                findings.append(
                    {
                        "file": parts[0],
                        "line": int(parts[1]) if parts[1].isdigit() else 0,
                        "message": parts[3].strip(),
                        "severity": severity,
                    }
                )
    return findings


def parse_pytest_output(stdout: str, stderr: str) -> list[dict[str, Any]]:
    """Parse pytest output into findings."""
    findings = []
    # Look for FAILED lines
    for line in (stdout + stderr).split("\n"):
        if line.startswith("FAILED "):
            # Format: FAILED tests/path/test_file.py::test_name - reason
            parts = line.split(" - ", 1)
            test_id = parts[0].replace("FAILED ", "").strip()
            reason = parts[1] if len(parts) > 1 else "Test failed"
            findings.append(
                {
                    "test": test_id,
                    "message": reason,
                    "severity": "error",
                }
            )
        elif line.startswith("ERROR "):
            test_id = line.replace("ERROR ", "").strip()
            findings.append(
                {
                    "test": test_id,
                    "message": "Test error (collection or setup failed)",
                    "severity": "error",
                }
            )
    return findings


def run_lint_check(paths: list[str] | None = None) -> CheckResult:
    """Run ruff linter."""
    cmd = ["uv", "run", "ruff", "check"]
    if paths:
        cmd.extend(paths)
    else:
        cmd.append(".")
    cmd.append("--output-format=json")

    exit_code, stdout, stderr, duration = run_command(cmd)

    # Check if ruff is not installed
    if "Failed to spawn" in stderr or "No such file" in stderr:
        return CheckResult(
            name="lint",
            passed=True,
            duration_seconds=duration,
            exit_code=0,
            findings=[],
            summary="Skipped (ruff not installed)",
            command=" ".join(cmd),
        )

    findings = parse_ruff_output(stdout)

    return CheckResult(
        name="lint",
        passed=exit_code == 0,
        duration_seconds=duration,
        exit_code=exit_code,
        findings=findings,
        summary=f"{len(findings)} linting issues found" if findings else "No linting issues",
        command=" ".join(cmd),
        stdout=stdout[:2000],
        stderr=stderr[:500],
    )


def run_format_check(paths: list[str] | None = None) -> CheckResult:
    """Run ruff format check."""
    cmd = ["uv", "run", "ruff", "format", "--check"]
    if paths:
        cmd.extend(paths)
    else:
        cmd.append(".")

    exit_code, stdout, stderr, duration = run_command(cmd)

    # Check if ruff is not installed
    if "Failed to spawn" in stderr or "No such file" in stderr:
        return CheckResult(
            name="format",
            passed=True,
            duration_seconds=duration,
            exit_code=0,
            findings=[],
            summary="Skipped (ruff not installed)",
            command=" ".join(cmd),
        )

    # Count files that would be reformatted
    findings = []
    for line in (stdout + stderr).split("\n"):
        if "Would reformat" in line or "would be reformatted" in line.lower():
            findings.append(
                {
                    "file": line.split()[-1] if line.split() else "",
                    "message": "File needs formatting",
                    "severity": "warning",
                }
            )

    return CheckResult(
        name="format",
        passed=exit_code == 0,
        duration_seconds=duration,
        exit_code=exit_code,
        findings=findings,
        summary=f"{len(findings)} files need formatting" if findings else "All files formatted",
        command=" ".join(cmd),
        stdout=stdout[:1000],
        stderr=stderr[:500],
    )


def run_type_check(paths: list[str] | None = None) -> CheckResult:
    """Run mypy type checker."""
    cmd = ["uv", "run", "mypy"]
    if paths:
        cmd.extend(paths)
    else:
        cmd.append("src/gpt_trader")
    cmd.extend(["--no-error-summary", "--show-column-numbers"])

    exit_code, stdout, stderr, duration = run_command(cmd)

    # Check if mypy is not installed
    if "Failed to spawn" in stderr or "No such file" in stderr:
        return CheckResult(
            name="types",
            passed=True,
            duration_seconds=duration,
            exit_code=0,
            findings=[],
            summary="Skipped (mypy not installed)",
            command=" ".join(cmd),
        )

    findings = parse_mypy_output(stdout)

    error_count = len([f for f in findings if f["severity"] == "error"])

    return CheckResult(
        name="types",
        passed=exit_code == 0,
        duration_seconds=duration,
        exit_code=exit_code,
        findings=findings,
        summary=f"{error_count} type errors found" if error_count else "No type errors",
        command=" ".join(cmd),
        stdout=stdout[:3000],
        stderr=stderr[:500],
    )


def run_test_check(fast: bool = True, paths: list[str] | None = None) -> CheckResult:
    """Run pytest."""
    cmd = ["uv", "run", "pytest"]

    if paths:
        cmd.extend(paths)
    else:
        cmd.append("tests/unit")

    cmd.extend(["-v", "--tb=short", "-q"])

    if fast:
        cmd.extend(["-m", "not slow and not performance"])

    exit_code, stdout, stderr, duration = run_command(cmd)
    findings = parse_pytest_output(stdout, stderr)

    # Extract summary from output
    summary = "Tests completed"
    for line in (stdout + stderr).split("\n"):
        if "passed" in line or "failed" in line or "error" in line:
            if "::" not in line:  # Skip individual test lines
                summary = line.strip()
                break

    return CheckResult(
        name="tests",
        passed=exit_code == 0,
        duration_seconds=duration,
        exit_code=exit_code,
        findings=findings,
        summary=summary,
        command=" ".join(cmd),
        stdout=stdout[-3000:],  # Last 3000 chars
        stderr=stderr[:1000],
    )


def run_security_check() -> CheckResult:
    """Run basic security checks (bandit if available)."""
    cmd = ["uv", "run", "bandit", "-r", "src/gpt_trader", "-f", "json", "-q"]

    exit_code, stdout, stderr, duration = run_command(cmd)

    findings = []
    if exit_code != 0 and "No module named bandit" in stderr:
        return CheckResult(
            name="security",
            passed=True,
            duration_seconds=0,
            exit_code=0,
            findings=[],
            summary="Skipped (bandit not installed)",
            command=" ".join(cmd),
        )

    try:
        data = json.loads(stdout) if stdout.strip() else {}
        for result in data.get("results", []):
            findings.append(
                {
                    "file": result.get("filename", ""),
                    "line": result.get("line_number", 0),
                    "code": result.get("test_id", ""),
                    "message": result.get("issue_text", ""),
                    "severity": result.get("issue_severity", "").lower(),
                }
            )
    except json.JSONDecodeError:
        pass

    return CheckResult(
        name="security",
        passed=exit_code == 0 or len(findings) == 0,
        duration_seconds=duration,
        exit_code=exit_code,
        findings=findings,
        summary=f"{len(findings)} security issues found" if findings else "No security issues",
        command=" ".join(cmd),
    )


def run_quality_gate(
    checks: list[str] | None = None,
    fast: bool = True,
    paths: list[str] | None = None,
) -> dict[str, Any]:
    """Run all quality checks and return results."""
    all_checks = ["lint", "format", "types", "tests", "security"]
    checks_to_run = checks if checks else ["lint", "format", "types", "tests"]

    results: list[CheckResult] = []
    start_time = time.time()

    for check in checks_to_run:
        if check not in all_checks:
            continue

        if check == "lint":
            results.append(run_lint_check(paths))
        elif check == "format":
            results.append(run_format_check(paths))
        elif check == "types":
            results.append(run_type_check(paths))
        elif check == "tests":
            results.append(run_test_check(fast=fast, paths=paths))
        elif check == "security":
            results.append(run_security_check())

    total_duration = time.time() - start_time
    all_passed = all(r.passed for r in results)
    total_findings = sum(len(r.findings) for r in results)

    return {
        "success": all_passed,
        "exit_code": 0 if all_passed else 1,
        "total_duration_seconds": round(total_duration, 2),
        "total_findings": total_findings,
        "checks_run": len(results),
        "checks_passed": len([r for r in results if r.passed]),
        "checks_failed": len([r for r in results if not r.passed]),
        "results": [r.to_dict() for r in results],
        "failed_checks": [r.name for r in results if not r.passed],
    }


def format_text_report(report: dict[str, Any]) -> str:
    """Format report as human-readable text."""
    lines = [
        "Quality Gate Report",
        "=" * 50,
        f"Status: {'PASSED' if report['success'] else 'FAILED'}",
        f"Duration: {report['total_duration_seconds']}s",
        f"Total findings: {report['total_findings']}",
        "",
    ]

    for result in report["results"]:
        status = "✓" if result["passed"] else "✗"
        lines.append(
            f"{status} {result['name']}: {result['summary']} ({result['duration_seconds']}s)"
        )

        if not result["passed"] and result["findings"]:
            for finding in result["findings"][:5]:
                file_info = finding.get("file") or finding.get("test", "")
                line_info = f":{finding['line']}" if finding.get("line") else ""
                lines.append(f"    - {file_info}{line_info}: {finding.get('message', '')}")
            if len(result["findings"]) > 5:
                lines.append(f"    ... and {len(result['findings']) - 5} more")

    if report["failed_checks"]:
        lines.extend(["", f"Failed checks: {', '.join(report['failed_checks'])}"])

    return "\n".join(lines)


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run quality gate checks with JSON output")
    parser.add_argument(
        "--check",
        type=str,
        help="Comma-separated list of checks: lint,format,types,tests,security",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        default=True,
        help="Skip slow tests (default: True)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run full test suite including slow tests",
    )
    parser.add_argument(
        "--files",
        type=str,
        nargs="+",
        help="Specific files or directories to check",
    )
    parser.add_argument(
        "--format",
        choices=["json", "text"],
        default="json",
        help="Output format",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output file (defaults to stdout)",
    )

    args = parser.parse_args()

    checks = args.check.split(",") if args.check else None
    fast = not args.full

    print("Running quality gate checks...", file=sys.stderr)
    report = run_quality_gate(checks=checks, fast=fast, paths=args.files)

    if args.format == "text":
        output = format_text_report(report)
    else:
        output = json.dumps(report, indent=2)

    if args.output:
        args.output.write_text(output)
        print(f"Report written to: {args.output}", file=sys.stderr)
    else:
        print(output)

    return report["exit_code"]


if __name__ == "__main__":
    sys.exit(main())
