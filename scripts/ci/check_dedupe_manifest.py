#!/usr/bin/env python3
"""Validate test deduplication manifest.

This script validates the dedupe_candidates.yaml manifest as part of CI.

Rules:
- Manifest must exist and be valid YAML
- Version must be 1
- No `status: in_progress` clusters older than 14 days without `pr_url`
- `expected_file_delta` must be present for all clusters

Usage:
    python scripts/ci/check_dedupe_manifest.py
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

MANIFEST_PATH = Path("tests/_triage/dedupe_candidates.yaml")
MAX_IN_PROGRESS_DAYS = 14


def load_manifest() -> dict[str, Any]:
    """Load and validate basic manifest structure."""
    if not MANIFEST_PATH.exists():
        # Manifest is optional until first generation
        return {}

    with open(MANIFEST_PATH) as f:
        manifest = yaml.safe_load(f)

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
