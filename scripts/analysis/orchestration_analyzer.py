#!/usr/bin/env python3
"""
Analyze orchestration layer dependencies and identify refactoring opportunities.

Usage:
    poetry run python scripts/analysis/orchestration_analyzer.py
"""

from __future__ import annotations

import ast
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ModuleInfo:
    """Information about a module."""

    name: str
    file_path: Path
    imports: list[str]
    classes: list[str]
    functions: list[str]
    line_count: int


def parse_module(file_path: Path) -> ModuleInfo:
    """Parse a Python module and extract metadata."""
    with open(file_path) as f:
        content = f.read()
        tree = ast.parse(content)

    # Extract imports from bot_v2.orchestration
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module and node.module.startswith("bot_v2.orchestration"):
                for alias in node.names:
                    imported = f"{node.module}.{alias.name}"
                    imports.append(imported)

    # Extract classes
    classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]

    # Extract top-level functions
    functions = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            functions.append(node.name)

    # Line count
    line_count = len(content.split("\n"))

    # Module name
    rel_path = file_path.relative_to(Path("src/bot_v2/orchestration"))
    module_name = str(rel_path).replace("/", ".").replace(".py", "")

    return ModuleInfo(
        name=module_name,
        file_path=file_path,
        imports=imports,
        classes=classes,
        functions=functions,
        line_count=line_count,
    )


def analyze_orchestration() -> dict[str, ModuleInfo]:
    """Analyze all orchestration modules."""
    orch_dir = Path("src/bot_v2/orchestration")
    modules = {}

    for py_file in orch_dir.rglob("*.py"):
        if py_file.name == "__init__.py":
            continue
        if "__pycache__" in str(py_file):
            continue

        module_info = parse_module(py_file)
        modules[module_info.name] = module_info

    return modules


def find_dependency_clusters(modules: dict[str, ModuleInfo]) -> dict[str, list[str]]:
    """Identify modules that frequently import each other (potential clusters)."""
    # Build dependency graph
    dep_graph = defaultdict(set)
    for mod_name, mod_info in modules.items():
        for imp in mod_info.imports:
            # Extract module name from import
            parts = imp.split(".")
            if len(parts) >= 4:  # bot_v2.orchestration.module_name.class
                dep_module = parts[2]  # Get module_name
                if dep_module in modules:
                    dep_graph[mod_name].add(dep_module)

    # Find clusters (modules that import each other)
    clusters = defaultdict(list)
    for mod_name, deps in dep_graph.items():
        # Check if any dependency also imports this module
        for dep in deps:
            if mod_name in dep_graph.get(dep, set()):
                cluster_key = tuple(sorted([mod_name, dep]))
                clusters[cluster_key].append("bidirectional")

    return dict(clusters)


def identify_hotspots(modules: dict[str, ModuleInfo]) -> list[tuple[str, int, int]]:
    """Identify complex modules (hotspots)."""
    hotspots = []

    for mod_name, mod_info in modules.items():
        # Complexity score: line count + class count * 100 + import count * 10
        score = mod_info.line_count + len(mod_info.classes) * 100 + len(mod_info.imports) * 10
        hotspots.append((mod_name, score, mod_info.line_count))

    return sorted(hotspots, key=lambda x: x[1], reverse=True)


def find_extraction_candidates(modules: dict[str, ModuleInfo]) -> list[str]:
    """Find modules that could be extracted to separate features."""
    candidates = []

    # Heuristics for extraction:
    # 1. Modules with specific domain focus (market_data, streaming, etc.)
    # 2. Modules with few dependencies on other orchestration modules
    # 3. Modules with clear single responsibility

    domain_patterns = [
        "market_data",
        "streaming",
        "telemetry",
        "monitor",
        "symbols",
        "equity",
    ]

    for mod_name, mod_info in modules.items():
        # Check if module has domain-specific focus
        for pattern in domain_patterns:
            if pattern in mod_name.lower():
                # Check dependency count
                orch_deps = [
                    imp
                    for imp in mod_info.imports
                    if "bot_v2.orchestration" in imp and "execution" not in imp
                ]
                if len(orch_deps) <= 3:  # Few orchestration dependencies
                    candidates.append(mod_name)
                    break

    return candidates


def generate_report(modules: dict[str, ModuleInfo]) -> str:
    """Generate markdown analysis report."""
    lines = [
        "# Orchestration Layer Analysis",
        "",
        "**Generated**: 2025-10-05",
        "**Purpose**: Identify refactoring opportunities in orchestration layer",
        "",
        "---",
        "",
        "## Summary",
        "",
        f"**Total Modules**: {len(modules)}",
        f"**Total Lines**: {sum(m.line_count for m in modules.values()):,}",
        f"**Total Classes**: {sum(len(m.classes) for m in modules.values())}",
        "",
        "### Complexity Indicators",
        "",
        f"- **Average Lines per Module**: {sum(m.line_count for m in modules.values()) // len(modules)}",
        f"- **Largest Module**: {max(modules.values(), key=lambda m: m.line_count).name} ({max(m.line_count for m in modules.values())} lines)",
        f"- **Most Imports**: {max(modules.values(), key=lambda m: len(m.imports)).name} ({max(len(m.imports) for m in modules.values())} imports)",
        "",
        "---",
        "",
        "## Hotspot Analysis",
        "",
        "Modules ranked by complexity (lines + classes*100 + imports*10):",
        "",
        "| Rank | Module | Complexity | Lines | Classes | Imports |",
        "|------|--------|------------|-------|---------|---------|",
    ]

    hotspots = identify_hotspots(modules)
    for i, (mod_name, score, line_count) in enumerate(hotspots[:15], 1):
        mod_info = modules[mod_name]
        lines.append(
            f"| {i} | `{mod_name}` | {score} | {line_count} | "
            f"{len(mod_info.classes)} | {len(mod_info.imports)} |"
        )

    lines.extend(
        [
            "",
            "---",
            "",
            "## Dependency Graph",
            "",
            "### Core Dependencies (Most Imported)",
            "",
        ]
    )

    # Count incoming dependencies
    incoming_deps = defaultdict(int)
    for mod_info in modules.values():
        for imp in mod_info.imports:
            parts = imp.split(".")
            if len(parts) >= 3:
                dep_module = parts[2]
                if dep_module in modules:
                    incoming_deps[dep_module] += 1

    sorted_deps = sorted(incoming_deps.items(), key=lambda x: x[1], reverse=True)[:10]
    for mod_name, count in sorted_deps:
        lines.append(f"- **{mod_name}**: imported by {count} modules")

    lines.extend(
        [
            "",
            "### Circular Dependencies",
            "",
        ]
    )

    clusters = find_dependency_clusters(modules)
    if clusters:
        for cluster_key, _ in clusters.items():
            lines.append(f"- `{cluster_key[0]}` ↔️ `{cluster_key[1]}` (bidirectional)")
    else:
        lines.append("✅ No circular dependencies detected")

    lines.extend(
        [
            "",
            "---",
            "",
            "## Extraction Candidates",
            "",
            "Modules that could be extracted to separate features:",
            "",
        ]
    )

    candidates = find_extraction_candidates(modules)
    for candidate in candidates:
        mod_info = modules[candidate]
        lines.append(
            f"### `{candidate}`"
            f"\n- **Lines**: {mod_info.line_count}"
            f"\n- **Dependencies**: {len(mod_info.imports)} orchestration imports"
            f"\n- **Reason**: Domain-specific, low coupling"
            f"\n"
        )

    lines.extend(
        [
            "---",
            "",
            "## Refactoring Recommendations",
            "",
            "### High Priority",
            "",
        ]
    )

    # Top 3 hotspots
    for i, (mod_name, score, line_count) in enumerate(hotspots[:3], 1):
        mod_info = modules[mod_name]
        lines.extend(
            [
                f"{i}. **{mod_name}** ({line_count} lines)",
                f"   - Split into smaller modules",
                f"   - Extract {len(mod_info.classes)} classes to separate files",
                "",
            ]
        )

    lines.extend(
        [
            "### Medium Priority",
            "",
            "- **Extract Domain Modules**: Move domain-specific modules to features/",
            "  - Market data → `features/market_data/`",
            "  - Streaming → `features/streaming/`",
            "  - Monitoring → Already in `monitoring/` (good!)",
            "",
            "- **Reduce Core Dependencies**: Modules with >5 orchestration imports should be reviewed",
            "",
            "### Low Priority",
            "",
            "- **Consolidate Similar Modules**: Consider merging small, related modules",
            "- **Documentation**: Add READMEs to execution/ subdir",
            "",
            "---",
            "",
            "## Proposed Structure (After Refactoring)",
            "",
            "```",
            "orchestration/",
            "├── core/                    # Core orchestration logic",
            "│   ├── bootstrap.py",
            "│   ├── lifecycle.py",
            "│   └── coordinator.py",
            "├── services/                # Service management",
            "│   ├── registry.py",
            "│   ├── rebinding.py",
            "│   └── telemetry.py",
            "├── execution/               # Order execution (existing)",
            "│   ├── ...",
            "├── strategy/                # Strategy orchestration",
            "│   ├── orchestrator.py",
            "│   ├── executor.py",
            "│   └── registry.py",
            "└── config/                  # Configuration",
            "    ├── controller.py",
            "    └── models.py",
            "",
            "# Extracted to features/",
            "features/",
            "├── market_data/",
            "├── streaming/",
            "└── guards/                  # Risk gates",
            "```",
            "",
            "---",
            "",
            "**Next Steps**: Create detailed refactoring plan in `docs/architecture/orchestration_refactor.md`",
        ]
    )

    return "\n".join(lines)


def main() -> int:
    """Main entry point."""
    print("Analyzing orchestration layer...")

    modules = analyze_orchestration()

    print(f"Found {len(modules)} modules")
    print("Generating report...")

    report = generate_report(modules)

    # Ensure output directory exists
    output_dir = Path("docs/architecture")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "orchestration_analysis.md"
    output_file.write_text(report)

    print(f"✅ Analysis written to: {output_file}")

    # Print summary
    hotspots = identify_hotspots(modules)
    print()
    print("=" * 60)
    print("Top 5 Hotspots:")
    print("=" * 60)
    for i, (mod_name, score, line_count) in enumerate(hotspots[:5], 1):
        print(f"  {i}. {mod_name}: {line_count} lines (score: {score})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
