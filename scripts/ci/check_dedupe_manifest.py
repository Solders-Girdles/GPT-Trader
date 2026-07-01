#!/usr/bin/env python3
"""Validate test deduplication manifest.

This script validates the dedupe_candidates.yaml manifest as part of CI.

Rules:
- Manifest must exist and be valid YAML
- Version must be 1
- No `status: in_progress` clusters older than 14 days without `pr_url`
- `expected_file_delta` must be present for all clusters
- `pending`/`in_progress` clusters must reference files that exist on disk
  (done clusters legitimately reference deleted files)

Usage:
    python scripts/ci/check_dedupe_manifest.py
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

MANIFEST_PATH = Path("tests/_triage/dedupe_candidates.yaml")
MAX_IN_PROGRESS_DAYS = 14

# A cluster-id key must be quoted if a YAML 1.1 loader (ruamel / the check-yaml
# pre-commit hook) would resolve it as a non-string: an integer (all-digit SHA
# prefix), a float (incl. hex ids like "310214e39620" that parse as inf), inf/nan,
# or a bool/null keyword. yaml.safe_load hides this (it reads them as strings), so
# we detect unquoted ambiguous keys from the raw text. This predicate is kept
# byte-for-byte identical to generate_dedupe_candidates._YAML_AMBIGUOUS_SCALAR
# (which quotes them on write); test_ambiguity_predicate_matches_generator enforces it.
_YAML_AMBIGUOUS_SCALAR = re.compile(
    r"""^(?:
        [-+]?[0-9][0-9_]*                                    # integer
        |[-+]?(?:[0-9][0-9_]*)?\.[0-9_]*(?:[eE][-+]?[0-9]+)?  # float with a dot
        |[-+]?[0-9][0-9_]*[eE][-+]?[0-9]+                     # float, exponent, no dot
        |[-+]?\.?(?:inf|nan)                                 # infinity / nan
        |true|false|yes|no|on|off|null|~                     # bool / null
    )$""",
    re.IGNORECASE | re.VERBOSE,
)
_UNQUOTED_CLUSTER_KEY = re.compile(r"^  ([^\s'\"#][^:]*):\s*$")


class _UniqueKeyLoader(yaml.SafeLoader):
    """SafeLoader that rejects duplicate mapping keys instead of silently
    keeping the last one."""


def _construct_mapping_no_duplicates(
    loader: _UniqueKeyLoader, node: yaml.MappingNode, deep: bool = False
) -> dict[Any, Any]:
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ValueError(f"Duplicate key in dedupe manifest: {key!r}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor("tag:yaml.org,2002:map", _construct_mapping_no_duplicates)


def load_manifest() -> dict[str, Any]:
    """Load and validate basic manifest structure."""
    if not MANIFEST_PATH.exists():
        # Manifest is optional until first generation
        return {}

    raw = MANIFEST_PATH.read_text(encoding="utf-8")

    ambiguous = sorted(
        {
            match.group(1)
            for line in raw.splitlines()
            if (match := _UNQUOTED_CLUSTER_KEY.match(line))
            and _YAML_AMBIGUOUS_SCALAR.match(match.group(1))
        }
    )
    if ambiguous:
        raise ValueError(
            f"Unquoted YAML-ambiguous cluster id(s) ({', '.join(ambiguous)}); "
            "regenerate with 'uv run agent-dedupe'"
        )

    manifest = yaml.load(raw, _UniqueKeyLoader)

    if not manifest:
        raise ValueError("Manifest is empty")

    if not isinstance(manifest, dict):
        raise ValueError("Manifest must be a mapping at the top level")

    version = manifest.get("version")
    if version != 1:
        raise ValueError(f"Manifest version must be 1 (got {version!r})")

    clusters = manifest.get("clusters", {})
    if not isinstance(clusters, dict):
        raise ValueError("Manifest `clusters` must be a mapping")

    return manifest


def validate_manifest(manifest: dict[str, Any]) -> list[str]:
    """Validate manifest content and return list of problems."""
    if not manifest:
        return []  # Empty/missing manifest is OK

    problems: list[str] = []
    clusters = manifest.get("clusters", {})

    for cluster_id, cluster in clusters.items():
        if not isinstance(cluster, dict):
            problems.append(f"Cluster {cluster_id}: must be a mapping")
            continue

        # Check required fields
        if "type" not in cluster:
            problems.append(f"Cluster {cluster_id}: missing 'type' field")

        if "files" not in cluster:
            problems.append(f"Cluster {cluster_id}: missing 'files' field")
        elif not isinstance(cluster.get("files"), list):
            problems.append(f"Cluster {cluster_id}: 'files' must be a list")

        if "decision" not in cluster:
            problems.append(f"Cluster {cluster_id}: missing 'decision' field")
        elif cluster["decision"] not in {"delete", "merge", "modernize"}:
            problems.append(f"Cluster {cluster_id}: invalid decision '{cluster['decision']}'")

        if "priority" not in cluster:
            problems.append(f"Cluster {cluster_id}: missing 'priority' field")
        elif cluster["priority"] not in {"high", "medium", "low"}:
            problems.append(f"Cluster {cluster_id}: invalid priority '{cluster['priority']}'")

        # Check expected_file_delta
        if "expected_file_delta" not in cluster:
            problems.append(f"Cluster {cluster_id}: missing 'expected_file_delta' field")

        # Check for stale in_progress clusters
        status = cluster.get("status")
        if status == "in_progress" and not cluster.get("pr_url"):
            added = cluster.get("added")
            if added:
                try:
                    added_date = datetime.strptime(added, "%Y-%m-%d")
                    days_old = (datetime.now() - added_date).days
                    if days_old > MAX_IN_PROGRESS_DAYS:
                        problems.append(
                            f"Cluster {cluster_id}: in_progress for {days_old} days "
                            f"without pr_url (max: {MAX_IN_PROGRESS_DAYS} days)"
                        )
                except ValueError:
                    problems.append(
                        f"Cluster {cluster_id}: invalid 'added' date format (expected YYYY-MM-DD)"
                    )

        # Validate status
        if status and status not in {"pending", "in_progress", "done"}:
            problems.append(f"Cluster {cluster_id}: invalid status '{status}'")

        if status == "in_progress" and not cluster.get("owner"):
            problems.append(f"Cluster {cluster_id}: in_progress cluster missing 'owner'")

        if status == "done":
            if not cluster.get("owner"):
                problems.append(f"Cluster {cluster_id}: done cluster missing 'owner'")
            if not cluster.get("pr_url"):
                problems.append(f"Cluster {cluster_id}: done cluster missing 'pr_url'")

        # Validate type
        cluster_type = cluster.get("type")
        if cluster_type and cluster_type not in {
            "source_fanout",
            "similar_names",
            "fixture_repeat",
        }:
            problems.append(f"Cluster {cluster_id}: invalid type '{cluster_type}'")

    return problems


def validate_cluster_files(manifest: dict[str, Any], root: Path | None = None) -> list[str]:
    """Check that unfinished clusters reference files that still exist.

    Pending and in_progress clusters describe work yet to happen, so every file
    they list must exist on disk. Done clusters are exempt: the dedupe work
    legitimately deleted their files, and the manifest keeps them as a durable
    audit trail.
    """
    if not manifest:
        return []

    base = root if root is not None else Path(".")
    problems: list[str] = []

    for cluster_id, cluster in manifest.get("clusters", {}).items():
        if not isinstance(cluster, dict):
            continue

        status = cluster.get("status", "pending")
        if status not in {"pending", "in_progress"}:
            continue

        files = cluster.get("files")
        if not isinstance(files, list):
            continue  # Shape problem already reported by validate_manifest

        missing = [
            file_path
            for file_path in files
            if isinstance(file_path, str) and not (base / file_path).exists()
        ]
        if missing:
            problems.append(
                f"Cluster {cluster_id}: {status} cluster references missing file(s): "
                f"{', '.join(missing)} (run 'uv run agent-dedupe' to refresh)"
            )

    return problems


def main(argv: list[str] | None = None) -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Validate test deduplication manifest.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if manifest does not exist",
    )
    args = parser.parse_args(argv)

    try:
        manifest = load_manifest()
    except ValueError as e:
        print(f"Dedupe manifest check failed: {e}", file=sys.stderr)
        return 2
    except yaml.YAMLError as e:
        print(f"Dedupe manifest parse error: {e}", file=sys.stderr)
        return 2

    if not manifest:
        if args.strict:
            print(f"Error: Manifest not found: {MANIFEST_PATH}", file=sys.stderr)
            print("Run: uv run agent-dedupe  # to generate", file=sys.stderr)
            return 1
        # Non-strict mode: missing manifest is OK
        return 0

    problems = validate_manifest(manifest)
    problems.extend(validate_cluster_files(manifest))

    if problems:
        print("Dedupe manifest validation issues:\n", file=sys.stderr)
        for problem in problems:
            print(f"- {problem}", file=sys.stderr)
        return 1

    cluster_count = len(manifest.get("clusters", {}))
    print(f"Dedupe manifest valid: {cluster_count} clusters")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
