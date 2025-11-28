#!/usr/bin/env python3
"""Generate module dependency graph for codebase navigation.

This tool analyzes import relationships between modules to:
- Build a dependency graph
- Detect circular dependencies
- Show what depends on a given module
- Show what a module depends on

Usage:
    python scripts/agents/dependency_graph.py [--format json|text|dot]
    python scripts/agents/dependency_graph.py --depends-on gpt_trader.cli
    python scripts/agents/dependency_graph.py --dependencies-of gpt_trader.cli
    python scripts/agents/dependency_graph.py --check-circular

Output:
    JSON dependency graph or specific query results.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).parent.parent.parent
SRC_DIR = PROJECT_ROOT / "src"


def extract_imports(file_path: Path) -> list[dict[str, Any]]:
    """Extract import statements from a Python file."""
    imports = []

    try:
        content = file_path.read_text()
        tree = ast.parse(content)
    except Exception:
        return imports

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append({
                    "type": "import",
                    "module": alias.name,
                    "alias": alias.asname,
                    "line": node.lineno,
                })
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                for alias in node.names:
                    imports.append({
                        "type": "from",
                        "module": node.module,
                        "name": alias.name,
                        "alias": alias.asname,
                        "level": node.level,
                        "line": node.lineno,
                    })

    return imports


def file_to_module(file_path: Path, base: Path) -> str:
    """Convert a file path to a module name."""
    relative = file_path.relative_to(base)
    parts = relative.with_suffix("").parts
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def build_dependency_graph(source_dir: Path) -> dict[str, Any]:
    """Build a complete dependency graph of the codebase."""
    graph: dict[str, set[str]] = defaultdict(set)
    reverse_graph: dict[str, set[str]] = defaultdict(set)
    module_files: dict[str, str] = {}
    external_deps: set[str] = set()

    # First pass: collect all internal modules
    internal_modules: set[str] = set()
    for py_file in source_dir.rglob("*.py"):
        if "__pycache__" in str(py_file):
            continue
        module = file_to_module(py_file, source_dir)
        internal_modules.add(module)
        module_files[module] = str(py_file.relative_to(PROJECT_ROOT))

    # Second pass: build dependency graph
    for py_file in source_dir.rglob("*.py"):
        if "__pycache__" in str(py_file):
            continue

        module = file_to_module(py_file, source_dir)
        imports = extract_imports(py_file)

        for imp in imports:
            imported_module = imp["module"]

            # Handle relative imports
            if imp.get("level", 0) > 0:
                # Convert relative to absolute
                parts = module.split(".")
                level = imp["level"]
                if level <= len(parts):
                    base_parts = parts[:-level] if level > 0 else parts
                    if imported_module:
                        imported_module = ".".join(base_parts + [imported_module])
                    else:
                        imported_module = ".".join(base_parts)

            # Check if it's an internal module
            is_internal = False
            for internal in internal_modules:
                if imported_module == internal or imported_module.startswith(internal + "."):
                    is_internal = True
                    # Normalize to the actual module
                    if imported_module in internal_modules:
                        graph[module].add(imported_module)
                        reverse_graph[imported_module].add(module)
                    elif internal in internal_modules:
                        graph[module].add(internal)
                        reverse_graph[internal].add(module)
                    break

            if not is_internal and not imported_module.startswith("."):
                # Track external dependencies
                top_level = imported_module.split(".")[0]
                external_deps.add(top_level)

    return {
        "modules": sorted(internal_modules),
        "module_files": module_files,
        "dependencies": {k: sorted(v) for k, v in graph.items() if v},
        "dependents": {k: sorted(v) for k, v in reverse_graph.items() if v},
        "external_dependencies": sorted(external_deps),
        "total_modules": len(internal_modules),
        "total_edges": sum(len(v) for v in graph.values()),
    }


def find_circular_dependencies(graph: dict[str, list[str]]) -> list[list[str]]:
    """Find circular dependencies in the graph."""
    cycles: list[list[str]] = []
    visited: set[str] = set()
    rec_stack: set[str] = set()
    path: list[str] = []

    def dfs(node: str) -> bool:
        visited.add(node)
        rec_stack.add(node)
        path.append(node)

        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                if dfs(neighbor):
                    return True
            elif neighbor in rec_stack:
                # Found a cycle
                cycle_start = path.index(neighbor)
                cycle = path[cycle_start:] + [neighbor]
                cycles.append(cycle)
                return True

        path.pop()
        rec_stack.remove(node)
        return False

    for node in graph:
        if node not in visited:
            dfs(node)

    return cycles


def get_dependencies_of(module: str, graph: dict[str, Any]) -> dict[str, Any]:
    """Get all dependencies of a module (what it imports)."""
    direct = graph["dependencies"].get(module, [])

    # Get transitive dependencies
    transitive: set[str] = set()
    to_visit = list(direct)
    visited: set[str] = set()

    while to_visit:
        current = to_visit.pop(0)
        if current in visited:
            continue
        visited.add(current)
        transitive.add(current)
        to_visit.extend(graph["dependencies"].get(current, []))

    return {
        "module": module,
        "direct_dependencies": direct,
        "transitive_dependencies": sorted(transitive - set(direct)),
        "total_direct": len(direct),
        "total_transitive": len(transitive),
    }


def get_dependents_of(module: str, graph: dict[str, Any]) -> dict[str, Any]:
    """Get all modules that depend on this module (what imports it)."""
    direct = graph["dependents"].get(module, [])

    # Get transitive dependents
    transitive: set[str] = set()
    to_visit = list(direct)
    visited: set[str] = set()

    while to_visit:
        current = to_visit.pop(0)
        if current in visited:
            continue
        visited.add(current)
        transitive.add(current)
        to_visit.extend(graph["dependents"].get(current, []))

    return {
        "module": module,
        "direct_dependents": direct,
        "transitive_dependents": sorted(transitive - set(direct)),
        "total_direct": len(direct),
        "total_transitive": len(transitive),
    }


def get_component_summary(graph: dict[str, Any]) -> dict[str, Any]:
    """Get a summary of components and their dependencies."""
    components: dict[str, set[str]] = defaultdict(set)

    # Group modules by top-level component
    for module in graph["modules"]:
        parts = module.split(".")
        if len(parts) >= 2:
            # gpt_trader.X or gpt_trader.features.X
            if parts[1] == "features" and len(parts) >= 3:
                component = parts[2]
            else:
                component = parts[1]
            components[component].add(module)

    # Calculate inter-component dependencies
    component_deps: dict[str, set[str]] = defaultdict(set)
    for src_module, deps in graph["dependencies"].items():
        src_parts = src_module.split(".")
        if len(src_parts) >= 2:
            if src_parts[1] == "features" and len(src_parts) >= 3:
                src_component = src_parts[2]
            else:
                src_component = src_parts[1]

            for dep in deps:
                dep_parts = dep.split(".")
                if len(dep_parts) >= 2:
                    if dep_parts[1] == "features" and len(dep_parts) >= 3:
                        dep_component = dep_parts[2]
                    else:
                        dep_component = dep_parts[1]

                    if src_component != dep_component:
                        component_deps[src_component].add(dep_component)

    return {
        "components": {k: len(v) for k, v in sorted(components.items())},
        "component_dependencies": {k: sorted(v) for k, v in sorted(component_deps.items())},
    }


def format_dot(graph: dict[str, Any]) -> str:
    """Format graph as DOT for visualization."""
    lines = [
        "digraph dependencies {",
        "  rankdir=LR;",
        "  node [shape=box];",
    ]

    # Group by component
    components: dict[str, list[str]] = defaultdict(list)
    for module in graph["modules"]:
        parts = module.split(".")
        if len(parts) >= 2:
            component = parts[1]
            components[component].append(module)

    # Create subgraphs for components
    for component, modules in sorted(components.items()):
        lines.append(f"  subgraph cluster_{component} {{")
        lines.append(f'    label="{component}";')
        for module in modules:
            short_name = module.split(".")[-1]
            lines.append(f'    "{module}" [label="{short_name}"];')
        lines.append("  }")

    # Add edges
    for src, deps in graph["dependencies"].items():
        for dep in deps:
            lines.append(f'  "{src}" -> "{dep}";')

    lines.append("}")
    return "\n".join(lines)


def format_text_report(graph: dict[str, Any]) -> str:
    """Format graph as human-readable text."""
    summary = get_component_summary(graph)

    lines = [
        "Module Dependency Graph",
        "=" * 50,
        f"Total Modules: {graph['total_modules']}",
        f"Total Dependencies: {graph['total_edges']}",
        f"External Dependencies: {len(graph['external_dependencies'])}",
        "",
        "Components:",
    ]

    for component, count in summary["components"].items():
        deps = summary["component_dependencies"].get(component, [])
        dep_str = f" -> {', '.join(deps)}" if deps else ""
        lines.append(f"  {component}: {count} modules{dep_str}")

    lines.extend([
        "",
        "External Dependencies:",
        f"  {', '.join(graph['external_dependencies'][:20])}",
    ])

    return "\n".join(lines)


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate module dependency graph"
    )
    parser.add_argument(
        "--format",
        choices=["json", "text", "dot"],
        default="json",
        help="Output format",
    )
    parser.add_argument(
        "--dependencies-of",
        type=str,
        help="Show dependencies of a specific module",
    )
    parser.add_argument(
        "--depends-on",
        type=str,
        help="Show what depends on a specific module",
    )
    parser.add_argument(
        "--check-circular",
        action="store_true",
        help="Check for circular dependencies",
    )
    parser.add_argument(
        "--component-summary",
        action="store_true",
        help="Show component-level summary",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output file",
    )

    args = parser.parse_args()

    print("Building dependency graph...", file=sys.stderr)
    graph = build_dependency_graph(SRC_DIR / "gpt_trader")

    # Handle specific queries
    if args.dependencies_of:
        result = get_dependencies_of(args.dependencies_of, graph)
        output = json.dumps(result, indent=2)
        print(output)
        return 0

    if args.depends_on:
        result = get_dependents_of(args.depends_on, graph)
        output = json.dumps(result, indent=2)
        print(output)
        return 0

    if args.check_circular:
        cycles = find_circular_dependencies(graph["dependencies"])
        result = {
            "has_circular": len(cycles) > 0,
            "cycle_count": len(cycles),
            "cycles": cycles[:10],  # Limit to first 10
        }
        output = json.dumps(result, indent=2)
        print(output)
        return 1 if cycles else 0

    if args.component_summary:
        result = get_component_summary(graph)
        output = json.dumps(result, indent=2)
        print(output)
        return 0

    # Full graph output
    if args.format == "text":
        output = format_text_report(graph)
    elif args.format == "dot":
        output = format_dot(graph)
    else:
        output = json.dumps(graph, indent=2)

    if args.output:
        args.output.write_text(output)
        print(f"Graph written to: {args.output}", file=sys.stderr)
    else:
        print(output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
