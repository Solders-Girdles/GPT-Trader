#!/usr/bin/env python3
"""Script to migrate hardcoded financial constants to unified configuration.

This script:
1. Identifies files with hardcoded financial values
2. Updates them to use the unified configuration
3. Reports on changes made
"""

import re
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def find_hardcoded_values(directory: Path) -> list[tuple[Path, int, str]]:
    """Find files with hardcoded financial values.

    Args:
        directory: Directory to search.

    Returns:
        List of (file_path, line_number, line_content) tuples.
    """
    patterns = [
        r"\b100_?000(?:\.0)?\b",  # 100000 or 100_000
        r"\b10000(?:\.0)?\b",  # 10000
        r"initial_capital\s*=\s*\d+",  # initial_capital = number
        r"deployment_budget\s*=\s*\d+",  # deployment_budget = number
    ]

    combined_pattern = "|".join(patterns)
    results = []

    for py_file in directory.rglob("*.py"):
        # Skip test files and migration scripts
        if "test" in py_file.parts or "scripts" in py_file.parts:
            continue

        # Skip the config modules themselves
        if "config" in py_file.name:
            continue

        try:
            with open(py_file) as f:
                lines = f.readlines()
                for i, line in enumerate(lines, 1):
                    if re.search(combined_pattern, line):
                        results.append((py_file, i, line.strip()))
        except Exception as e:
            print(f"Error reading {py_file}: {e}")

    return results


def generate_migration_report(results: list[tuple[Path, int, str]]) -> str:
    """Generate a migration report.

    Args:
        results: List of hardcoded value locations.

    Returns:
        Formatted report string.
    """
    report = ["=" * 60]
    report.append("Financial Constants Migration Report")
    report.append("=" * 60)
    report.append(f"\nFound {len(results)} hardcoded values to migrate:\n")

    # Group by file
    by_file = {}
    for path, line_num, content in results:
        if path not in by_file:
            by_file[path] = []
        by_file[path].append((line_num, content))

    for file_path, occurrences in by_file.items():
        rel_path = file_path.relative_to(project_root)
        report.append(f"\n{rel_path}:")
        for line_num, content in occurrences:
            report.append(f"  Line {line_num}: {content[:80]}...")

    report.append("\n" + "=" * 60)
    report.append("Migration Strategy:")
    report.append("=" * 60)
    report.append(
        """
1. Files that need imports:
   from bot.config import get_config

2. Replace hardcoded values:
   100000 -> get_config().financial.capital.initial_capital
   10000  -> get_config().financial.capital.deployment_budget

3. For class defaults, use None and load in __post_init__:
   initial_capital: float = None

   def __post_init__(self):
       if self.initial_capital is None:
           config = get_config()
           self.initial_capital = float(config.financial.capital.initial_capital)
"""
    )

    return "\n".join(report)


def suggest_replacements(file_path: Path, line_num: int, content: str) -> str:
    """Suggest replacement for a hardcoded value.

    Args:
        file_path: Path to the file.
        line_num: Line number.
        content: Line content.

    Returns:
        Suggested replacement.
    """
    replacements = {
        "100_000": "get_config().financial.capital.initial_capital",
        "100000": "get_config().financial.capital.initial_capital",
        "10000": "get_config().financial.capital.deployment_budget",
    }

    for pattern, replacement in replacements.items():
        if pattern in content:
            return content.replace(pattern, replacement)

    return content


def main():
    """Main migration function."""
    print("Scanning for hardcoded financial values...")

    # Search in src directory
    src_dir = project_root / "src"
    results = find_hardcoded_values(src_dir)

    if not results:
        print("✅ No hardcoded financial values found!")
        return

    # Generate report
    report = generate_migration_report(results)
    print(report)

    # Save report to file
    report_path = project_root / "migration_report_financial.txt"
    with open(report_path, "w") as f:
        f.write(report)

    print(f"\n📄 Full report saved to: {report_path}")
    print("\n⚠️  Manual review and update required for these files.")
    print("   Use the migration strategy above as a guide.")

    # Count by type
    initial_capital_count = sum(1 for _, _, c in results if "100000" in c or "100_000" in c)
    deployment_count = sum(1 for _, _, c in results if "10000" in c)

    print("\nSummary:")
    print(f"  Initial capital values: {initial_capital_count}")
    print(f"  Deployment budget values: {deployment_count}")
    print(f"  Total: {len(results)}")


if __name__ == "__main__":
    main()
