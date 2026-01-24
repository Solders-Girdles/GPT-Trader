#!/usr/bin/env python3
"""Generate an environment variable reference for AI agents.

The goal is to provide a low-maintenance, code-driven inventory of environment
variables that affect runtime behavior.

Inputs:
- Static scan of python sources for os.getenv(...) and parse_*_env(...) calls
- Optional enrichment from config/environments/.env.template (defaults/comments)

Output (deterministic):
- var/agents/configuration/environment_variables.json
- var/agents/configuration/environment_variables.md
- var/agents/configuration/index.json
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOTS = [
    PROJECT_ROOT / "src" / "gpt_trader",
    PROJECT_ROOT / "scripts",
]
ENV_TEMPLATE_PATH = PROJECT_ROOT / "config" / "environments" / ".env.template"

ENV_READER_FUNCTIONS = {
    "parse_bool_env",
    "parse_int_env",
    "parse_float_env",
    "parse_decimal_env",
    "parse_list_env",
    "parse_mapping_env",
    "get_env_bool",
    "get_env_int",
    "get_env_float",
    "get_env_decimal",
    "parse_env_list",
    "parse_env_mapping",
    "coerce_env_value",
    "require_env_value",
    "_env_flag",
}

ENV_MULTI_KEY_READERS = {
    "_env_lookup",
}

SPECIAL_PREFIX_READERS: dict[str, dict[str, str]] = {
    "src/gpt_trader/app/config/bot_config.py": {
        "_risk_env": "RISK_",
        "_risk_int": "RISK_",
        "_risk_decimal": "RISK_",
        "_risk_float": "RISK_",
        "_health_float": "HEALTH_",
        "_health_int": "HEALTH_",
    },
    "src/gpt_trader/features/live_trade/risk/config.py": {
        "_get_env": "RISK_",
    },
}


@dataclass(frozen=True)
class CodeReference:
    path: str
    line: int
    reader: str


@dataclass(frozen=True)
class TemplateReference:
    line: int
    default: str
    comment: str
    commented_out: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate environment variable reference.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "var" / "agents" / "configuration",
        help="Output directory for reference files (default: var/agents/configuration).",
    )
    return parser.parse_args()


def _extract_module_string_constants(tree: ast.Module) -> dict[str, str]:
    constants: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            if not isinstance(node.value.value, str):
                continue
            for target in node.targets:
                if isinstance(target, ast.Name):
                    constants[target.id] = node.value.value
        if isinstance(node, ast.AnnAssign) and isinstance(node.value, ast.Constant):
            if not isinstance(node.value.value, str):
                continue
            if isinstance(node.target, ast.Name):
                constants[node.target.id] = node.value.value
    return constants


def _resolve_string_literal(node: ast.AST, constants: dict[str, str]) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        return constants.get(node.id)
    return None


def _resolve_default_literal(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, (str, int, float, bool)):
        return str(node.value)
    return None


def _call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _call_name(node.value)
        if parent:
            return f"{parent}.{node.attr}"
        return None
    return None


def _iter_env_var_names_from_args(
    args: list[ast.AST],
    constants: dict[str, str],
) -> list[str]:
    names: list[str] = []
    for arg in args:
        name = _resolve_string_literal(arg, constants)
        if not name:
            continue
        if not re.fullmatch(r"[A-Z0-9_]+", name):
            continue
        names.append(name)
    return names


def collect_code_references() -> dict[str, list[CodeReference]]:
    references: dict[str, list[CodeReference]] = {}

    for root in SRC_ROOTS:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.py")):
            rel_path = path.relative_to(PROJECT_ROOT).as_posix()
            if rel_path.startswith("scripts/agents/"):
                continue

            try:
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=rel_path)
            except SyntaxError:
                continue

            constants = _extract_module_string_constants(tree)
            special_readers = SPECIAL_PREFIX_READERS.get(rel_path, {})

            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue

                name: str | None = None
                reader: str | None = None

                call_name = _call_name(node.func)
                if call_name == "os.getenv" and node.args:
                    name = _resolve_string_literal(node.args[0], constants)
                    reader = "os.getenv"
                elif call_name == "os.environ.get" and node.args:
                    name = _resolve_string_literal(node.args[0], constants)
                    reader = "os.environ.get"
                elif call_name in ENV_MULTI_KEY_READERS:
                    names = _iter_env_var_names_from_args(node.args, constants)
                    if names:
                        for resolved in names:
                            location = CodeReference(
                                path=rel_path,
                                line=getattr(node, "lineno", 0) or 0,
                                reader=call_name,
                            )
                            references.setdefault(resolved, []).append(location)
                    continue
                elif call_name in ENV_READER_FUNCTIONS and node.args:
                    name = _resolve_string_literal(node.args[0], constants)
                    reader = call_name
                elif (
                    isinstance(node.func, ast.Name)
                    and node.func.id in special_readers
                    and node.args
                ):
                    key = _resolve_string_literal(node.args[0], constants)
                    if key:
                        name = f"{special_readers[node.func.id]}{key}"
                        reader = node.func.id

                if not name or not re.fullmatch(r"[A-Z0-9_]+", name):
                    continue

                location = CodeReference(
                    path=rel_path,
                    line=getattr(node, "lineno", 0) or 0,
                    reader=reader or "unknown",
                )
                references.setdefault(name, []).append(location)

    return references


def parse_env_template() -> dict[str, list[TemplateReference]]:
    entries: dict[str, list[TemplateReference]] = {}
    if not ENV_TEMPLATE_PATH.exists():
        return entries

    for index, raw_line in enumerate(
        ENV_TEMPLATE_PATH.read_text(encoding="utf-8").splitlines(), start=1
    ):
        stripped = raw_line.strip()
        if not stripped:
            continue

        commented_out = stripped.startswith("#")
        candidate = stripped.lstrip("#").strip() if commented_out else stripped
        if "=" not in candidate:
            continue

        left, right = candidate.split("=", maxsplit=1)
        var_name = left.strip()
        if not re.fullmatch(r"[A-Z0-9_]+", var_name):
            continue

        value_part, _, comment_part = right.partition("#")
        default = value_part.strip()
        comment = comment_part.strip()
        entry = TemplateReference(
            line=index,
            default=default,
            comment=comment,
            commented_out=commented_out,
        )
        entries.setdefault(var_name, []).append(entry)

    return entries


def _unique_paths(references: list[CodeReference]) -> list[str]:
    paths = sorted({ref.path for ref in references})
    return paths


def render_markdown(
    *,
    variables: list[dict[str, Any]],
    code_only: list[str],
    template_only: list[str],
) -> str:
    lines: list[str] = []
    lines.append("# Environment Variable Reference")
    lines.append("")
    lines.append("Generated by `scripts/agents/generate_environment_variable_reference.py`.")
    lines.append("")
    lines.append(
        "| Env var | In `.env.template` | Used in code | Template default | Code locations |"
    )
    lines.append(
        "|--------|---------------------|--------------|------------------|----------------|"
    )

    for item in variables:
        name = item["name"]
        in_template = "yes" if item["in_env_template"] else "no"
        used_in_code = "yes" if item["code_references"] else "no"
        template_default = item.get("template_default", "")
        locations = ", ".join(item.get("code_paths", []))
        lines.append(
            f"| `{name}` | {in_template} | {used_in_code} | `{template_default}` | {locations} |"
        )

    lines.append("")

    if code_only:
        lines.append("## Code-only variables")
        lines.append("")
        lines.append(
            "These appear in code but are not present in `config/environments/.env.template`."
        )
        lines.append("")
        for name in code_only:
            lines.append(f"- `{name}`")
        lines.append("")

    if template_only:
        lines.append("## Template-only variables")
        lines.append("")
        lines.append(
            "These appear in `config/environments/.env.template` but were not found in code scans."
        )
        lines.append("")
        for name in template_only:
            lines.append(f"- `{name}`")
        lines.append("")

    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    code_refs = collect_code_references()
    template_refs = parse_env_template()

    all_names = sorted(set(code_refs) | set(template_refs))

    variables: list[dict[str, Any]] = []
    code_only: list[str] = []
    template_only: list[str] = []

    for name in all_names:
        in_template = name in template_refs
        in_code = name in code_refs

        if in_code and not in_template:
            code_only.append(name)
        if in_template and not in_code:
            template_only.append(name)

        template_default = ""
        template_locations: list[dict[str, Any]] = []
        if in_template:
            template_locations = [
                {
                    "line": ref.line,
                    "default": ref.default,
                    "comment": ref.comment,
                    "commented_out": ref.commented_out,
                }
                for ref in sorted(template_refs[name], key=lambda ref: ref.line)
            ]
            template_default = template_locations[0]["default"]

        code_locations: list[dict[str, Any]] = []
        code_paths: list[str] = []
        if in_code:
            refs = sorted(code_refs[name], key=lambda ref: (ref.path, ref.line, ref.reader))
            code_locations = [
                {"path": ref.path, "line": ref.line, "reader": ref.reader} for ref in refs
            ]
            code_paths = _unique_paths(refs)

        variables.append(
            {
                "name": name,
                "in_env_template": in_template,
                "template_references": template_locations,
                "template_default": template_default,
                "code_references": code_locations,
                "code_paths": code_paths,
            }
        )

    payload = {
        "version": "1.0",
        "env_template_path": ENV_TEMPLATE_PATH.relative_to(PROJECT_ROOT).as_posix(),
        "variables": variables,
        "summary": {
            "total": len(variables),
            "code_only": len(code_only),
            "template_only": len(template_only),
        },
    }

    (output_dir / "environment_variables.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "environment_variables.md").write_text(
        render_markdown(variables=variables, code_only=code_only, template_only=template_only),
        encoding="utf-8",
    )
    (output_dir / "index.json").write_text(
        json.dumps(
            {
                "version": "1.0",
                "description": "Environment variable reference (static scan + .env.template enrichment).",
                "files": {
                    "environment_variables_json": "environment_variables.json",
                    "environment_variables_markdown": "environment_variables.md",
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
