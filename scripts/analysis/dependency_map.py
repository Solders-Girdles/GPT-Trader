"""Dependency analysis tooling for the gpt_trader codebase.

This script parses the Python modules under ``src`` and builds an import
graph.  It emits JSON reports describing the dependency structure, highlights
cycles, and lists potential dead/entry modules.  Optionally it can generate a
GraphViz ``.dot`` file for visualization.
"""

from __future__ import annotations

import argparse
import ast
import json
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from collections.abc import Iterable, Iterator


DEFAULT_SOURCE_ROOT = Path("src")
DEFAULT_PACKAGE = "gpt_trader"


@dataclass
class ModuleNode:
    name: str
    path: Path
    imports: set[str]


def discover_modules(root: Path, package: str) -> dict[str, ModuleNode]:
    modules: dict[str, ModuleNode] = {}
    for path in root.rglob("*.py"):
        if path.name == "__pycache__":
            continue
        try:
            rel = path.relative_to(root)
        except ValueError:
            continue
        parts = rel.with_suffix("").parts
        if parts and parts[0] == package:
            module = ".".join(parts)
        else:
            module = package + "." + ".".join(parts)
        module = normalize_module_name(module)
        modules[module] = ModuleNode(name=module, path=path, imports=set())
    return modules


def parse_imports(modules: dict[str, ModuleNode], package: str) -> None:
    for module in modules.values():
        try:
            source = module.path.read_text(encoding="utf-8")
        except Exception:
            continue
        try:
            tree = ast.parse(source, filename=str(module.path))
        except SyntaxError:
            continue

        visitor = ImportVisitor(current_module=module.name)
        visitor.visit(tree)
        for target in visitor.imports:
            normalized = normalize_module_name(target)
            if normalized.startswith(package):
                module.imports.add(normalized)


class ImportVisitor(ast.NodeVisitor):
    def __init__(self, current_module: str) -> None:
        self.current_module = current_module
        self.imports: set[str] = set()

    def visit_Import(self, node: ast.Import) -> None:  # noqa: N802
        for alias in node.names:
            self.imports.add(alias.name)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:  # noqa: N802
        module = node.module or ""
        if node.level and self.current_module:
            resolved = resolve_relative_import(
                current_module=self.current_module,
                level=node.level,
                module=module,
            )
        else:
            resolved = module
        if resolved:
            self.imports.add(resolved)


def resolve_relative_import(current_module: str, level: int, module: str) -> str:
    parts = current_module.split(".")
    if parts[-1] == "__init__":
        package_parts = parts[:-1]
    else:
        package_parts = parts[:-1]

    remove = level - 1
    if remove > len(package_parts):
        base_parts: list[str] = []
    else:
        base_parts = package_parts[: len(package_parts) - remove]
    if module:
        base_parts.extend(module.split("."))
    return ".".join(part for part in base_parts if part)


def normalize_module_name(module: str) -> str:
    module = module.replace("/", ".")
    module = module.replace("\\", ".")
    if module.endswith(".__init__"):
        module = module[: -len(".__init__")]
    return module


def build_dependency_graph(modules: dict[str, ModuleNode]) -> dict[str, set[str]]:
    graph = {module: set() for module in modules}
    for module, node in modules.items():
        for dep in node.imports:
            if dep in modules and dep != module:
                graph[module].add(dep)
    return graph


def invert_graph(graph: dict[str, set[str]]) -> dict[str, set[str]]:
    reverse: dict[str, set[str]] = {node: set() for node in graph}
    for src, deps in graph.items():
        for dest in deps:
            reverse.setdefault(dest, set()).add(src)
    return reverse


def strongly_connected_components(graph: dict[str, set[str]]) -> list[list[str]]:
    index = 0
    stack: list[str] = []
    indices: dict[str, int] = {}
    lowlink: dict[str, int] = {}
    on_stack: set[str] = set()
    sccs: list[list[str]] = []

    def strongconnect(node: str) -> None:
        nonlocal index
        indices[node] = index
        lowlink[node] = index
        index += 1
        stack.append(node)
        on_stack.add(node)

        for neighbor in graph.get(node, ()):  # type: ignore[arg-type]
            if neighbor not in indices:
                strongconnect(neighbor)
                lowlink[node] = min(lowlink[node], lowlink[neighbor])
            elif neighbor in on_stack:
                lowlink[node] = min(lowlink[node], indices[neighbor])

        if lowlink[node] == indices[node]:
            component: list[str] = []
            while True:
                w = stack.pop()
                on_stack.remove(w)
                component.append(w)
                if w == node:
                    break
            sccs.append(component)

    for node in graph:
        if node not in indices:
            strongconnect(node)

    return sccs


def detect_cycles(graph: dict[str, set[str]]) -> list[list[str]]:
    sccs = strongly_connected_components(graph)
    return [sorted(comp) for comp in sccs if len(comp) > 1]


def top_n_by_degree(degrees: dict[str, int], n: int = 10) -> list[tuple[str, int]]:
    return sorted(degrees.items(), key=lambda item: item[1], reverse=True)[:n]


def analyze(graph: dict[str, set[str]]) -> dict[str, object]:
    reverse = invert_graph(graph)
    out_degree = {node: len(neighbors) for node, neighbors in graph.items()}
    in_degree = {node: len(reverse.get(node, set())) for node in graph}

    cycles = detect_cycles(graph)
    entry_points = [node for node, indeg in in_degree.items() if indeg == 0]
    leaf_modules = [node for node, outdeg in out_degree.items() if outdeg == 0]

    return {
        "module_count": len(graph),
        "edge_count": sum(len(neigh) for neigh in graph.values()),
        "top_out_degree": top_n_by_degree(out_degree),
        "top_in_degree": top_n_by_degree(in_degree),
        "entry_points": sorted(entry_points),
        "leaf_modules": sorted(leaf_modules),
        "cycles": cycles,
    }


def emit_dot(graph: dict[str, set[str]], path: Path, focus: set[str] | None = None) -> None:
    focus = focus or set(graph.keys())
    with path.open("w", encoding="utf-8") as fh:
        fh.write("digraph dependencies {\n")
        fh.write("  rankdir=LR;\n")
        for node in sorted(focus):
            fh.write(f'  "{node}";\n')
        for src, dests in graph.items():
            if src not in focus:
                continue
            for dest in dests:
                if dest in focus:
                    fh.write(f'  "{src}" -> "{dest}";\n')
        fh.write("}\n")


def apply_focus(graph: dict[str, set[str]], focus_modules: Iterable[str]) -> set[str]:
    focus_set = set()
    queue = deque(focus_modules)
    seen = set()

    while queue:
        module = queue.popleft()
        if module in seen or module not in graph:
            continue
        seen.add(module)
        focus_set.add(module)
        for dep in graph[module]:
            queue.append(dep)
    return focus_set


def map_tests_to_modules(
    patterns: Iterable[str], modules: dict[str, ModuleNode], package: str
) -> dict[str, set[str]]:
    tests: dict[str, set[str]] = {}
    for pattern in patterns:
        for path in Path().glob(pattern):
            if not path.is_file():
                continue
            try:
                source = path.read_text(encoding="utf-8")
            except Exception:
                continue
            try:
                tree = ast.parse(source, filename=str(path))
            except SyntaxError:
                continue

            visitor = ImportVisitor(current_module=str(path))
            visitor.visit(tree)

            imported_modules = set()
            for target in visitor.imports:
                normalized = normalize_module_name(target)
                if normalized.startswith(package):
                    # retain only modules from our codebase
                    imported_modules.add(normalized)

            if imported_modules:
                tests[str(path)] = imported_modules

    return tests


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze project dependencies")
    parser.add_argument("--root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--package", type=str, default=DEFAULT_PACKAGE)
    parser.add_argument("--output", type=Path, help="Write JSON report to this path")
    parser.add_argument("--dot", type=Path, help="Write GraphViz dot graph to this path")
    parser.add_argument(
        "--focus",
        nargs="+",
        help="Limit graph/export to modules starting with these prefixes",
    )
    parser.add_argument(
        "--tests",
        nargs="+",
        help="Glob patterns pointing to test files to analyse",
    )
    parser.add_argument("--show", action="store_true", help="Print JSON report to stdout")
    args = parser.parse_args()

    modules = discover_modules(args.root, args.package)
    parse_imports(modules, args.package)
    graph = build_dependency_graph(modules)

    test_map = {}
    if args.tests:
        test_map = map_tests_to_modules(args.tests, modules, args.package)

    report = analyze(graph)
    report["graph"] = {module: sorted(deps) for module, deps in graph.items()}
    report["module_to_path"] = {module: str(node.path) for module, node in sorted(modules.items())}
    if test_map:
        report["tests_to_modules"] = {test: sorted(deps) for test, deps in sorted(test_map.items())}

    if args.focus:
        focus = set()
        for prefix in args.focus:
            focus.update(module for module in graph if module.startswith(prefix))
        focus = apply_focus(graph, focus)
    else:
        focus = None

    if args.dot:
        emit_dot(graph, args.dot, focus)

    if args.output:
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if args.show or not args.output:
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
