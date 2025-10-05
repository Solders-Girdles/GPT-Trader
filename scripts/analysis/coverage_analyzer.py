#!/usr/bin/env python3
"""
Analyze coverage data and generate per-slice heatmap.

Usage:
    poetry run python scripts/analysis/coverage_analyzer.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class CoverageStats:
    """Coverage statistics for a module."""

    file_path: str
    statements: int
    executed: int
    missing: int
    coverage_pct: float


@dataclass
class SliceCoverage:
    """Coverage statistics for a feature slice."""

    slice_name: str
    total_statements: int
    total_executed: int
    total_missing: int
    coverage_pct: float
    modules: list[CoverageStats]

    @property
    def status_icon(self) -> str:
        """Get status icon based on coverage."""
        if self.coverage_pct >= 90:
            return "🟢"
        elif self.coverage_pct >= 70:
            return "🟡"
        else:
            return "🔴"


def parse_coverage_json(coverage_json_path: Path) -> dict[str, CoverageStats]:
    """Parse coverage.json and extract per-file stats."""
    with open(coverage_json_path) as f:
        data = json.load(f)

    file_stats = {}
    files = data.get("files", {})

    for file_path, file_data in files.items():
        summary = file_data.get("summary", {})
        num_statements = summary.get("num_statements", 0)
        covered_lines = summary.get("covered_lines", 0)
        missing_lines = summary.get("missing_lines", 0)

        if num_statements > 0:
            coverage_pct = (covered_lines / num_statements) * 100
        else:
            coverage_pct = 0.0

        file_stats[file_path] = CoverageStats(
            file_path=file_path,
            statements=num_statements,
            executed=covered_lines,
            missing=missing_lines,
            coverage_pct=coverage_pct,
        )

    return file_stats


def group_by_slice(file_stats: dict[str, CoverageStats]) -> dict[str, SliceCoverage]:
    """Group coverage stats by feature slice."""
    slice_data = defaultdict(lambda: {"statements": 0, "executed": 0, "missing": 0, "modules": []})

    for file_path, stats in file_stats.items():
        # Extract slice name from path
        if "src/bot_v2/features/" in file_path:
            parts = file_path.split("src/bot_v2/features/")[1].split("/")
            slice_name = f"features/{parts[0]}"
        elif "src/bot_v2/orchestration/" in file_path:
            slice_name = "orchestration"
        elif "src/bot_v2/monitoring/" in file_path:
            slice_name = "monitoring"
        elif "src/bot_v2/state/" in file_path:
            slice_name = "state"
        elif "src/bot_v2/cli/" in file_path:
            slice_name = "cli"
        elif "src/bot_v2/" in file_path:
            # Other top-level modules
            parts = file_path.split("src/bot_v2/")[1].split("/")
            slice_name = parts[0] if len(parts) > 1 else "core"
        else:
            slice_name = "other"

        slice_data[slice_name]["statements"] += stats.statements
        slice_data[slice_name]["executed"] += stats.executed
        slice_data[slice_name]["missing"] += stats.missing
        slice_data[slice_name]["modules"].append(stats)

    # Convert to SliceCoverage objects
    slices = {}
    for slice_name, data in slice_data.items():
        total_statements = data["statements"]
        total_executed = data["executed"]
        coverage_pct = (total_executed / total_statements * 100) if total_statements > 0 else 0.0

        slices[slice_name] = SliceCoverage(
            slice_name=slice_name,
            total_statements=total_statements,
            total_executed=total_executed,
            total_missing=data["missing"],
            coverage_pct=coverage_pct,
            modules=sorted(data["modules"], key=lambda m: m.coverage_pct),
        )

    return slices


def generate_heatmap_markdown(slices: dict[str, SliceCoverage]) -> str:
    """Generate markdown heatmap report."""
    lines = [
        "# Coverage Heatmap",
        "",
        "**Generated**: 2025-10-05",
        "**Purpose**: Identify testing gaps per feature slice",
        "",
        "---",
        "",
        "## Summary",
        "",
    ]

    # Overall stats
    total_statements = sum(s.total_statements for s in slices.values())
    total_executed = sum(s.total_executed for s in slices.values())
    overall_coverage = (total_executed / total_statements * 100) if total_statements > 0 else 0

    lines.extend(
        [
            f"**Overall Coverage**: {overall_coverage:.2f}%",
            f"**Total Statements**: {total_statements:,}",
            f"**Executed**: {total_executed:,}",
            f"**Missing**: {total_statements - total_executed:,}",
            "",
            "### Coverage Status Legend",
            "",
            "- 🟢 **Excellent** (≥90%): Well-tested, minimal gaps",
            "- 🟡 **Good** (70-89%): Adequate coverage, some gaps",
            "- 🔴 **Needs Work** (<70%): Significant testing gaps",
            "",
            "---",
            "",
            "## Per-Slice Coverage",
            "",
            "| Slice | Coverage | Statements | Executed | Missing | Status |",
            "|-------|----------|------------|----------|---------|--------|",
        ]
    )

    # Sort slices by coverage (lowest first to highlight gaps)
    sorted_slices = sorted(slices.values(), key=lambda s: s.coverage_pct)

    for slice_coverage in sorted_slices:
        lines.append(
            f"| {slice_coverage.slice_name} | "
            f"{slice_coverage.coverage_pct:.1f}% | "
            f"{slice_coverage.total_statements:,} | "
            f"{slice_coverage.total_executed:,} | "
            f"{slice_coverage.total_missing:,} | "
            f"{slice_coverage.status_icon} |"
        )

    lines.extend(["", "---", "", "## Detailed Slice Analysis", ""])

    # Detailed analysis for slices with <80% coverage
    for slice_coverage in sorted_slices:
        if slice_coverage.coverage_pct < 80:
            lines.extend(
                [
                    f"### {slice_coverage.status_icon} {slice_coverage.slice_name} ({slice_coverage.coverage_pct:.1f}%)",
                    "",
                    "**Priority**: HIGH - Needs immediate attention",
                    "",
                    "#### Lowest Coverage Modules",
                    "",
                    "| Module | Coverage | Statements | Missing |",
                    "|--------|----------|------------|---------|",
                ]
            )

            # Show worst 5 modules
            worst_modules = slice_coverage.modules[:5]
            for module in worst_modules:
                rel_path = (
                    module.file_path.split("src/bot_v2/")[1]
                    if "src/bot_v2/" in module.file_path
                    else module.file_path
                )
                lines.append(
                    f"| {rel_path} | "
                    f"{module.coverage_pct:.1f}% | "
                    f"{module.statements} | "
                    f"{module.missing} |"
                )

            lines.extend(["", ""])

    # Testing recommendations
    lines.extend(
        [
            "---",
            "",
            "## Testing Backlog (Priority Order)",
            "",
            "Based on coverage gaps and module criticality:",
            "",
        ]
    )

    priority_slices = [s for s in sorted_slices if s.coverage_pct < 80]
    for i, slice_coverage in enumerate(priority_slices[:5], 1):
        gap_count = len([m for m in slice_coverage.modules if m.coverage_pct < 70])
        lines.append(
            f"{i}. **{slice_coverage.slice_name}** - {slice_coverage.coverage_pct:.1f}% "
            f"({gap_count} modules below 70%)"
        )

    lines.extend(
        [
            "",
            "---",
            "",
            "## Recommended Actions",
            "",
            "### Immediate (This Sprint)",
            "",
        ]
    )

    # Find critical low-coverage areas
    critical_areas = []
    for slice_name, slice_coverage in slices.items():
        if slice_coverage.coverage_pct < 60 and "features" in slice_name:
            critical_areas.append(slice_name)

    if critical_areas:
        lines.extend(
            [
                "1. **Critical Coverage Gaps** - Add tests for:",
                "",
            ]
        )
        for area in critical_areas:
            lines.append(f"   - `{area}` - {slices[area].coverage_pct:.1f}% coverage")
        lines.append("")

    lines.extend(
        [
            "2. **Establish Baseline Tests**",
            "   - Each module should have at least 70% coverage",
            "   - Focus on happy path + error handling",
            "",
            "3. **Integration Tests**",
            "   - Build scenario tests for broker/orchestration interactions",
            "   - Use recorded Coinbase fixtures for deterministic tests",
            "",
            "### Phase 1 Goals",
            "",
            "- [ ] Bring all feature slices to ≥80% coverage",
            "- [ ] Add integration tests for orchestration layer",
            "- [ ] Establish coverage gates in CI (fail if coverage drops)",
            "",
            "### Phase 2 Goals",
            "",
            "- [ ] Achieve ≥90% coverage across all features",
            "- [ ] Add property-based tests for critical calculations (risk, fees)",
            "- [ ] Build comprehensive scenario test suite",
            "",
            "---",
            "",
            "**Next Steps**: See `docs/testing/coverage_backlog.md` for detailed module-level tasks",
        ]
    )

    return "\n".join(lines)


def main() -> int:
    """Main entry point."""
    coverage_json = Path("coverage.json")

    if not coverage_json.exists():
        print("ERROR: coverage.json not found")
        print("Run: poetry run pytest --cov=src/bot_v2 --cov-report=json")
        return 1

    print("Parsing coverage data...")
    file_stats = parse_coverage_json(coverage_json)

    print(f"Analyzing {len(file_stats)} files...")
    slices = group_by_slice(file_stats)

    print(f"Generating heatmap for {len(slices)} slices...")
    markdown = generate_heatmap_markdown(slices)

    # Ensure output directory exists
    output_dir = Path("docs/testing")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "coverage_heatmap.md"
    output_file.write_text(markdown)

    print(f"✅ Coverage heatmap written to: {output_file}")

    # Print summary
    total_statements = sum(s.total_statements for s in slices.values())
    total_executed = sum(s.total_executed for s in slices.values())
    overall_coverage = (total_executed / total_statements * 100) if total_statements > 0 else 0

    print()
    print("=" * 60)
    print(f"Overall Coverage: {overall_coverage:.2f}%")
    print("=" * 60)

    # Show top 3 gaps
    sorted_slices = sorted(slices.values(), key=lambda s: s.coverage_pct)
    print("\nTop 3 Coverage Gaps:")
    for i, slice_coverage in enumerate(sorted_slices[:3], 1):
        print(
            f"  {i}. {slice_coverage.slice_name}: {slice_coverage.coverage_pct:.1f}% "
            f"({slice_coverage.total_missing} statements missing)"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
