#!/usr/bin/env python3
"""Lightweight legacy-pattern scanner for repo hygiene."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

TEXT_EXTENSIONS = {
    ".md",
    ".py",
    ".sh",
    ".txt",
    ".toml",
    ".yml",
    ".yaml",
    ".env",
    ".ini",
    ".cfg",  # naming: allow
}

SKIP_DIRS = {
    ".git",
    ".venv",
    "__pycache__",
    "node_modules",
    "runtime_data",
    "logs",
    "var",
    "data",
    "secrets",
    ".uv-cache",
}

LEGACY_ENV_VARS = {
    "COINBASE_ENABLE_DERIVATIVES": {
        "allowed_files": {
            "docs/DEPRECATIONS.md",
            "docs/LEGACY_DEBT_WORKLIST.md",
            "scripts/ci/check_legacy_patterns.py",
            "src/gpt_trader/app/config/bot_config.py",
            "src/gpt_trader/preflight/checks/environment.py",
            "src/gpt_trader/preflight/context.py",
            "src/gpt_trader/monitoring/configuration_guardian/environment.py",
        },
        "allowed_prefixes": ("tests/",),
    },
}

REQUESTS_CALL_ATTRS = {
    "get",
    "post",
    "put",
    "delete",
    "head",
    "options",
    "patch",
    "request",
    "Session",
}


def _iter_text_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*"):
        if path.is_dir():
            if path.name in SKIP_DIRS:
                continue
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.name == "Makefile" or path.suffix in TEXT_EXTENSIONS:
            files.append(path)
    return files


def _check_deprecated_env_usage(files: list[Path]) -> list[str]:
    errors: list[str] = []
    for path in files:
        rel_path = path.relative_to(REPO_ROOT).as_posix()
        text = path.read_text(encoding="utf-8", errors="ignore")
        for env_var, allow in LEGACY_ENV_VARS.items():
            if env_var not in text:
                continue
            if rel_path in allow["allowed_files"]:
                continue
            if rel_path.startswith(allow["allowed_prefixes"]):
                continue
            errors.append(f"{rel_path}: legacy env var '{env_var}' referenced outside allowlist")
    return errors


def _collect_import_aliases(tree: ast.AST) -> tuple[set[str], set[str], set[str]]:
    time_aliases: set[str] = set()
    sleep_aliases: set[str] = set()
    requests_aliases: set[str] = set()
    requests_direct_calls: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "time":
                    time_aliases.add(alias.asname or alias.name)
                if alias.name == "requests":
                    requests_aliases.add(alias.asname or alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module == "time":
                for alias in node.names:
                    if alias.name == "sleep":
                        sleep_aliases.add(alias.asname or alias.name)
            if node.module == "requests":
                for alias in node.names:
                    requests_direct_calls.add(alias.asname or alias.name)

    return time_aliases, sleep_aliases, requests_aliases | requests_direct_calls


def _check_blocking_calls_in_async(src_root: Path) -> list[str]:
    errors: list[str] = []
    for path in src_root.rglob("*.py"):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue

        time_aliases, sleep_aliases, requests_aliases = _collect_import_aliases(tree)
        if not (time_aliases or sleep_aliases or requests_aliases):
            continue

        for node in ast.walk(tree):
            if not isinstance(node, ast.AsyncFunctionDef):
                continue
            for call in ast.walk(node):
                if not isinstance(call, ast.Call):
                    continue
                func = call.func
                if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
                    if func.value.id in time_aliases and func.attr == "sleep":
                        errors.append(
                            f"{path.relative_to(REPO_ROOT)}:{call.lineno}: time.sleep inside async def"
                        )
                    if func.value.id in requests_aliases and func.attr in REQUESTS_CALL_ATTRS:
                        errors.append(
                            f"{path.relative_to(REPO_ROOT)}:{call.lineno}: requests.{func.attr} inside async def"
                        )
                if isinstance(func, ast.Name):
                    if func.id in sleep_aliases:
                        errors.append(
                            f"{path.relative_to(REPO_ROOT)}:{call.lineno}: sleep inside async def"
                        )
                    if func.id in requests_aliases:
                        errors.append(
                            f"{path.relative_to(REPO_ROOT)}:{call.lineno}: requests call inside async def"
                        )
    return errors


def _check_duplicate_deploy_entrypoints() -> list[str]:
    errors: list[str] = []
    allowed_compose = {
        "deploy/gpt_trader/docker/docker-compose.yaml",
        "deploy/gpt_trader/docker/docker-compose.infrastructure.yaml",
    }
    for path in REPO_ROOT.rglob("docker-compose*.y*ml"):
        rel_path = path.relative_to(REPO_ROOT).as_posix()
        if rel_path not in allowed_compose:
            errors.append(f"{rel_path}: unexpected docker-compose file")

    for path in (REPO_ROOT / "deploy").rglob("kubernetes"):
        if path.is_dir():
            rel_path = path.relative_to(REPO_ROOT).as_posix()
            errors.append(f"{rel_path}: legacy kubernetes deployment directory detected")

    return errors


def main() -> int:
    errors: list[str] = []
    files = _iter_text_files(REPO_ROOT)
    errors.extend(_check_deprecated_env_usage(files))
    errors.extend(_check_blocking_calls_in_async(REPO_ROOT / "src"))
    errors.extend(_check_duplicate_deploy_entrypoints())

    if errors:
        print("Legacy pattern check failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1
    print("Legacy pattern check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
