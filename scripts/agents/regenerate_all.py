#!/usr/bin/env python3
"""Regenerate all static context files in var/agents/.

This script runs all generator tools to refresh the static context files
that are committed to the repository for AI agent consumption.

Usage:
    python scripts/agents/regenerate_all.py           # Regenerate all
    python scripts/agents/regenerate_all.py --verify  # Check freshness only
    python scripts/agents/regenerate_all.py --list    # List generators

Entry point:
    uv run agent-regenerate
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Final
from typing import NamedTuple

PROJECT_ROOT = Path(__file__).parent.parent.parent

# Map of generator scripts to their output directories
# Format: (script_name, output_subdir, description)
GENERATORS: list[tuple[str, str, str]] = [
    ("generate_config_schemas.py", "schemas", "Configuration schemas"),
    ("export_model_schemas.py", "models", "Domain model schemas"),
    ("generate_event_catalog.py", "logging", "Event catalog"),
    ("generate_test_inventory.py", "testing", "Test inventory"),
    ("generate_validator_registry.py", "validation", "Validator registry"),
    ("generate_broker_api_docs.py", "broker", "Broker API docs"),
    ("generate_reasoning_artifacts.py", "reasoning", "Reasoning artifacts"),
    ("generate_agent_health_schema.py", "health", "Agent health schema"),
]


class GeneratorResult(NamedTuple):
    """Result of running a generator script."""

    script: str
    success: bool
    duration: float
    output_dir: str
    error: str | None = None


_NORMALIZE_SUFFIXES: Final[frozenset[str]] = frozenset(
    {".dot", ".json", ".md", ".txt", ".yaml", ".yml"}
)


def _normalize_text_file(path: Path) -> bool:
    """Match repo hygiene hooks for generated files.

    - Strip trailing whitespace (spaces/tabs) per line.
    - Ensure file ends with a single newline (unless empty).
    """
    if path.suffix not in _NORMALIZE_SUFFIXES:
        return False
    try:
        original = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return False

    normalized_lines = [line.rstrip(" \t") for line in original.splitlines()]
    normalized = "\n".join(normalized_lines)
    if normalized:
        normalized += "\n"

    if normalized == original:
        return False

    path.write_text(normalized, encoding="utf-8")
    return True


def normalize_output_dir(output_dir: Path) -> None:
    if not output_dir.exists():
        return
    for candidate in output_dir.rglob("*"):
        if candidate.is_file():
            _normalize_text_file(candidate)


def run_generator(
    script_name: str,
    output_dir: str,
    extra_args: Sequence[str] = (),
) -> GeneratorResult:
    """Run a single generator script.

    Args:
        script_name: Name of the script file
        output_dir: Expected output subdirectory

    Returns:
        GeneratorResult with success status and timing
    """
    scripts_dir = PROJECT_ROOT / "scripts" / "agents"
    script_path = scripts_dir / script_name

    if not script_path.exists():
        return GeneratorResult(
            script=script_name,
            success=False,
            duration=0.0,
            output_dir=output_dir,
            error=f"Script not found: {script_path}",
        )

    start = time.time()
    try:
        command = [sys.executable, str(script_path), *extra_args]
        result = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=120,  # 2 minute timeout per generator
        )
        duration = time.time() - start

        if result.returncode != 0:
            error_msg = result.stderr.strip() or result.stdout.strip() or "Unknown error"
            return GeneratorResult(
                script=script_name,
                success=False,
                duration=duration,
                output_dir=output_dir,
                error=error_msg[:500],  # Truncate long errors
            )

        normalize_output_dir(PROJECT_ROOT / "var" / "agents" / output_dir)

        return GeneratorResult(
            script=script_name,
            success=True,
            duration=duration,
            output_dir=output_dir,
        )

    except subprocess.TimeoutExpired:
        return GeneratorResult(
            script=script_name,
            success=False,
            duration=120.0,
            output_dir=output_dir,
            error="Timed out after 120 seconds",
        )
    except Exception as e:
        return GeneratorResult(
            script=script_name,
            success=False,
            duration=time.time() - start,
            output_dir=output_dir,
            error=str(e),
        )


def regenerate_all(
    verbose: bool = True,
    generators: Sequence[tuple[str, str, str]] | None = None,
    *,
    reasoning_validate: bool = False,
    reasoning_strict: bool = False,
) -> tuple[list[GeneratorResult], bool]:
    """Run all generators.

    Args:
        verbose: Print progress to stderr
        generators: Optional list of generator tuples to run

    Returns:
        Tuple of (results list, overall success)
    """
    results: list[GeneratorResult] = []
    var_agents = PROJECT_ROOT / "var" / "agents"
    generators_to_run = list(generators) if generators is not None else GENERATORS

    # Ensure output directory exists
    var_agents.mkdir(parents=True, exist_ok=True)

    for script_name, output_dir, description in generators_to_run:
        if verbose:
            print(f"Running {script_name}... ", end="", flush=True, file=sys.stderr)

        extra_args: list[str] = []
        if script_name == "generate_reasoning_artifacts.py":
            if reasoning_validate or reasoning_strict:
                extra_args.append("--validate")
            if reasoning_strict:
                extra_args.append("--strict")

        result = run_generator(script_name, output_dir, extra_args)
        results.append(result)

        if verbose:
            if result.success:
                print(f"OK ({result.duration:.1f}s)", file=sys.stderr)
            else:
                print("FAILED", file=sys.stderr)
                if result.error:
                    # Print first line of error
                    error_line = result.error.split("\n")[0]
                    print(f"  Error: {error_line}", file=sys.stderr)

    success = all(r.success for r in results)
    return results, success


def verify_freshness(generators: Sequence[tuple[str, str, str]] | None = None) -> int:
    """Verify that generated files are up-to-date.

    Runs all generators and checks if there are uncommitted changes
    in var/agents/. Returns 0 if fresh, 1 if stale.
    """
    import shutil

    var_agents = PROJECT_ROOT / "var" / "agents"
    generators_to_run = list(generators) if generators is not None else GENERATORS

    # Check if git is available
    if shutil.which("git") is None:
        print("Error: git not found", file=sys.stderr)
        return 1

    print("Regenerating context files...", file=sys.stderr)
    results, success = regenerate_all(
        verbose=True,
        generators=generators_to_run,
        reasoning_validate=True,
        reasoning_strict=True,
    )

    if not success:
        print("\nSome generators failed. Cannot verify freshness.", file=sys.stderr)
        return 1

    if not generators_to_run:
        print("\nNo generators selected; nothing to verify.", file=sys.stderr)
        return 1

    diff_paths = [str(var_agents / output_dir) for _, output_dir, _ in generators_to_run]

    # Check for git changes
    result = subprocess.run(
        ["git", "diff", "--quiet", "--", *diff_paths],
        cwd=PROJECT_ROOT,
        capture_output=True,
    )

    if result.returncode != 0:
        print("\n" + "=" * 50, file=sys.stderr)
        print("STALE: Agent context files have changed!", file=sys.stderr)
        print("=" * 50, file=sys.stderr)
        print("\nRun 'uv run agent-regenerate' and commit the changes.", file=sys.stderr)

        # Show what changed
        diff_result = subprocess.run(
            ["git", "diff", "--stat", "--", *diff_paths],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
        )
        if diff_result.stdout:
            print("\nChanged files:", file=sys.stderr)
            print(diff_result.stdout, file=sys.stderr)

        return 1

    print("\nAll context files are up-to-date.", file=sys.stderr)
    return 0


def list_generators() -> None:
    """List all available generators."""
    print("Available generators:")
    print("-" * 96)
    print(f"{'Key':<12} {'Script':<35} {'Description':<30} Status")
    print("-" * 96)
    for script_name, output_dir, description in GENERATORS:
        script_path = PROJECT_ROOT / "scripts" / "agents" / script_name
        status = "OK" if script_path.exists() else "MISSING"
        print(f"  {output_dir:<12} {script_name:<35} {description:<30} [{status}]")
    print("-" * 96)
    print("Keys map to output subdirectories under var/agents/")


def parse_only_arg(only_arg: str | None) -> set[str]:
    if not only_arg:
        return set()
    return {item.strip() for item in only_arg.split(",") if item.strip()}


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Regenerate all agent context files in var/agents/",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    %(prog)s                 # Regenerate all context files
    %(prog)s --verify        # Check if files are up-to-date (for CI)
    %(prog)s --list          # List available generators
    %(prog)s --only testing  # Regenerate one generator
    %(prog)s --quiet         # Suppress progress output
        """,
    )
    parser.add_argument(
        "--only",
        type=str,
        help=(
            "Comma-separated list of generators to run (e.g., 'schemas,testing'). "
            "Valid values: schemas, models, logging, testing, validation, broker, reasoning, health"
        ),
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify freshness only (returns non-zero if stale)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available generators and exit",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )

    args = parser.parse_args()

    if args.list:
        list_generators()
        return 0

    requested = parse_only_arg(args.only)
    if args.only and not requested:
        print("Error: --only provided but no generator keys found.", file=sys.stderr)
        list_generators()
        return 2

    valid_keys = {output_dir for _, output_dir, _ in GENERATORS}
    invalid = sorted(requested - valid_keys)
    if invalid:
        print(f"Error: Unknown generator key(s): {', '.join(invalid)}", file=sys.stderr)
        list_generators()
        return 2

    generators_to_run = [g for g in GENERATORS if g[1] in requested] if requested else GENERATORS

    if args.verify:
        return verify_freshness(generators=generators_to_run)

    # Normal regeneration
    results, success = regenerate_all(
        verbose=not args.quiet,
        generators=generators_to_run,
        reasoning_validate=True,
    )

    # Summary
    passed = len([r for r in results if r.success])
    failed = len([r for r in results if not r.success])
    total_time = sum(r.duration for r in results)

    if not args.quiet:
        print("-" * 50, file=sys.stderr)
        print(f"Completed: {passed} passed, {failed} failed ({total_time:.1f}s)", file=sys.stderr)

        if success:
            print("\nAll generators completed successfully.", file=sys.stderr)
        else:
            print("\nSome generators failed. Check errors above.", file=sys.stderr)
            for r in results:
                if not r.success:
                    print(f"  - {r.script}: {r.error}", file=sys.stderr)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
