#!/usr/bin/env python3
"""
CI check to ensure no deprecated ConfigLoader usage is introduced.

This script fails if it finds any usage of deprecated ConfigLoader functions
in production code (excluding tests, __init__.py, and documentation).
"""

import sys
from pathlib import Path
import subprocess
from typing import List


DEPRECATED_PATTERNS = [
    "ConfigLoader(",
    "get_config_loader(",
    "from gpt_trader.config import.*get_config",  # Specific import pattern
    "set_config_loader(",
    "with_config(",
]

# Files and directories to exclude from the check
EXCLUDE_PATHS = {
    "tests/",
    "test_",
    "__pycache__/",
    ".pytest_cache/",
    "venv/",
    ".venv/",
    "__init__.py",  # Keep deprecated exports for backward compatibility
    ".md:",
    "docs/",  # Documentation files
    "scripts/ci/check_config_deprecated_usage.py",  # This script itself
    "src/gpt_trader/features/brokerages/coinbase/client/base.py",  # Intentional fallback for test compatibility
    "src/gpt_trader/orchestration/perps_bot_builder.py",  # Has legit with_config method
    "src/gpt_trader/orchestration/perps_bot.py",  # Has legit _align_registry_with_config method
}


def check_for_deprecated_usage() -> list[str]:
    """Check for deprecated ConfigLoader usage in the codebase.

    Returns:
        List of findings with file paths and line numbers
    """
    findings = []

    # Use git grep to search for deprecated patterns
    for pattern in DEPRECATED_PATTERNS:
        try:
            result = subprocess.run(
                ["git", "grep", "-n", pattern], capture_output=True, text=True, check=True
            )

            lines = result.stdout.strip().split("\n")
            for line in lines:
                if not line.strip():
                    continue

                # Parse the git grep output (filename:line_number:content)
                if ":" not in line:
                    continue

                parts = line.split(":", 2)
                if len(parts) < 3:
                    continue

                file_path = parts[0]

                # Check if this file/path should be excluded
                should_exclude = False
                for exclude_path in EXCLUDE_PATHS:
                    if exclude_path in file_path or file_path.startswith(exclude_path):
                        should_exclude = True
                        break

                if not should_exclude:
                    findings.append(line)

        except subprocess.CalledProcessError:
            # git grep returns non-zero exit code when no matches found
            # That's actually OK for our purposes
            continue

    return findings


def main() -> int:
    """Main function that returns exit code for CI."""
    print("🔍 Checking for deprecated ConfigLoader usage...")

    findings = check_for_deprecated_usage()

    if not findings:
        print("✅ No deprecated ConfigLoader usage found!")
        print(
            "✅ All configuration should use ConfigManager from gpt_trader.orchestration.configuration.manager"
        )
        return 0

    print(f"❌ Found {len(findings)} instances of deprecated ConfigLoader usage:")
    print()

    for finding in findings:
        print(f"  📍 {finding}")

    print()
    print("Please replace deprecated ConfigLoader usage with ConfigManager:")
    print("  - Use ConfigManager from gpt_trader.orchestration.configuration.manager")
    print("  - Use BotConfig.from_profile() instead of get_config()")
    print("  - See tests/integration/test_config_migration.py for examples")

    return 1


if __name__ == "__main__":
    sys.exit(main())
