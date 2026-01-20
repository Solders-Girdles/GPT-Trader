#!/usr/bin/env python3
"""Generate test deduplication candidates manifest.

This script analyzes the test suite to identify clusters of test files
that could benefit from consolidation, merging, or cleanup.

Detection heuristics:
- Source Fanout: Source modules imported by many test files (>threshold)
- Similar Names: Test files with common naming patterns in the same directory

Usage:
    python scripts/agents/generate_dedupe_candidates.py           # Generate/update manifest
    python scripts/agents/generate_dedupe_candidates.py --verify  # Check freshness (for CI)
    python scripts/agents/generate_dedupe_candidates.py --cluster abc123
    python scripts/agents/generate_dedupe_candidates.py --stats
    python scripts/agents/generate_dedupe_candidates.py --next-pr

Output:
    tests/_triage/dedupe_candidates.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).parent.parent.parent
SOURCE_TEST_MAP_PATH = PROJECT_ROOT / "var" / "agents" / "testing" / "source_test_map.json"
MANIFEST_PATH = PROJECT_ROOT / "tests" / "_triage" / "dedupe_candidates.yaml"

# Detection thresholds
SOURCE_FANOUT_THRESHOLD = 5
SIMILAR_NAME_THRESHOLD = 3


def generate_cluster_id(files: list[str], source_modules: list[str]) -> str:
    """Generate deterministic cluster ID from sorted files and modules."""
    content = "|".join(sorted(files)) + "||" + "|".join(sorted(source_modules))
    return hashlib.sha256(content.encode()).hexdigest()[:12]


def load_source_test_map() -> dict[str, Any]:
    """Load the source-to-test mapping."""
    if not SOURCE_TEST_MAP_PATH.exists():
        print(f"Error: Source test map not found: {SOURCE_TEST_MAP_PATH}", file=sys.stderr)
        print("Run: uv run agent-tests  # to regenerate", file=sys.stderr)
        sys.exit(1)

    with open(SOURCE_TEST_MAP_PATH) as f:
        return json.load(f)


def load_existing_manifest() -> dict[str, Any]:
    """Load existing manifest for merge mode."""
    if not MANIFEST_PATH.exists():
        return {}

    with open(MANIFEST_PATH) as f:
        return yaml.safe_load(f) or {}


def detect_source_fanout(
    source_test_map: dict[str, Any],
    threshold: int = SOURCE_FANOUT_THRESHOLD,
) -> list[dict[str, Any]]:
    """Detect source modules imported by many test files."""
    clusters = []
    source_to_tests = source_test_map.get("source_to_tests", {})

    for module in sorted(source_to_tests):
        test_files = source_to_tests[module]
        # Only consider unit tests for deduplication
        unit_files = [f for f in test_files if f.startswith("tests/unit/")]

        if len(unit_files) > threshold:
            cluster_id = generate_cluster_id(unit_files, [module])
            clusters.append(
                {
                    "id": cluster_id,
                    "type": "source_fanout",
                    "description": f"{len(unit_files)} test files import {module}",
                    "files": sorted(unit_files),
                    "source_modules": [module],
                    "file_count": len(unit_files),
                }
            )

    return clusters


def detect_similar_names(source_test_map: dict[str, Any]) -> list[dict[str, Any]]:
    """Detect test files with common naming patterns in the same directory."""
    clusters = []

    # Collect all unit test files
    all_test_files: list[str] = []
    for test_files in source_test_map.get("source_to_tests", {}).values():
        all_test_files.extend(f for f in test_files if f.startswith("tests/unit/"))
    all_test_files = sorted(set(all_test_files))

    # Group by directory
    files_by_dir: dict[str, list[str]] = defaultdict(list)
    for test_file in all_test_files:
        directory = str(Path(test_file).parent)
        files_by_dir[directory].append(test_file)

    # Within each directory, find common base patterns
    for directory in sorted(files_by_dir):
        files = sorted(files_by_dir[directory])
        if len(files) < SIMILAR_NAME_THRESHOLD:
            continue

        # Extract base patterns (first 3 words after test_)
        pattern_groups: dict[str, list[str]] = defaultdict(list)
        for file_path in files:
            filename = Path(file_path).stem  # test_risk_manager_daily_pnl
            # Remove test_ prefix and split by underscores
            if filename.startswith("test_"):
                parts = filename[5:].split("_")
                # Use first 2 words as pattern (e.g., "risk_manager")
                if len(parts) >= 2:
                    pattern = "_".join(parts[:2])
                    pattern_groups[pattern].append(file_path)

        # Create clusters for patterns with multiple files
        for pattern in sorted(pattern_groups):
            pattern_files = pattern_groups[pattern]
            if len(pattern_files) >= SIMILAR_NAME_THRESHOLD:
                cluster_id = generate_cluster_id(pattern_files, [])
                clusters.append(
                    {
                        "id": cluster_id,
                        "type": "similar_names",
                        "description": f"{len(pattern_files)} test files match pattern test_{pattern}_* in {directory}",
                        "files": sorted(pattern_files),
                        "source_modules": [],
                        "file_count": len(pattern_files),
                        "pattern": f"test_{pattern}_*",
                    }
                )

    return clusters


def infer_decision(cluster: dict[str, Any]) -> str:
    """Infer suggested decision based on cluster characteristics."""
    # Currently all clusters are recommended for merge
    # Future: could use cluster["type"] or file_count to differentiate
    _ = cluster.get("file_count", 0)  # Reserved for future decision logic
    return "merge"


def infer_priority(cluster: dict[str, Any]) -> str:
    """Infer priority based on cluster characteristics."""
    file_count = cluster.get("file_count", 0)

    if file_count > 10:
        return "high"
    if file_count > 5:
        return "medium"
    return "low"


def infer_target_location(cluster: dict[str, Any]) -> str | None:
    """Infer target location for merged tests."""
    files = cluster.get("files", [])
    if not files:
        return None

    # Use the directory of the first file
    directory = str(Path(files[0]).parent)

    # For similar_names, use the pattern
    if cluster.get("type") == "similar_names":
        pattern = cluster.get("pattern", "")
        if pattern:
            base_name = pattern.replace("_*", "").replace("*", "")
            return f"{directory}/{base_name}.py"

    # For source_fanout, use the module name
    source_modules = cluster.get("source_modules", [])
    if source_modules:
        module_name = source_modules[0].split(".")[-1]
        return f"{directory}/test_{module_name}.py"

    return None


def build_manifest(
    clusters: list[dict[str, Any]],
    existing_manifest: dict[str, Any],
) -> dict[str, Any]:
    """Build the manifest, preserving existing status fields."""
    existing_clusters = existing_manifest.get("clusters", {})

    pending_clusters: list[tuple[str, dict[str, Any]]] = []
    by_priority = {"high": 0, "medium": 0, "low": 0}
    by_decision = {"delete": 0, "merge": 0, "modernize": 0}

    for cluster in clusters:
        cluster_id = cluster["id"]

        # Preserve existing status fields if cluster exists
        existing = existing_clusters.get(cluster_id, {})

        decision = existing.get("decision") or infer_decision(cluster)
        priority = existing.get("priority") or infer_priority(cluster)
        target_location = existing.get("target_location") or infer_target_location(cluster)

        file_count = cluster.get("file_count", len(cluster.get("files", [])))

        new_cluster = {
            "type": cluster["type"],
            "description": cluster["description"],
            "files": cluster["files"],
            "source_modules": cluster.get("source_modules", []),
            "decision": decision,
            "target_location": target_location,
            "expected_test_delta": existing.get("expected_test_delta"),
            "expected_file_delta": existing.get("expected_file_delta", -(file_count - 1)),
            "priority": priority,
            "rationale": existing.get("rationale")
            or f"Consolidate {file_count} test files into single parametrized file",
            # Preserved fields
            "status": existing.get("status", "pending"),
            "owner": existing.get("owner"),
            "pr_url": existing.get("pr_url"),
            "added": existing.get("added") or datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        }

        # Remove None values for cleaner YAML
        new_cluster = {k: v for k, v in new_cluster.items() if v is not None}

        pending_clusters.append((cluster_id, new_cluster))

        by_priority[priority] = by_priority.get(priority, 0) + 1
        by_decision[decision] = by_decision.get(decision, 0) + 1

    priority_order = {"high": 0, "medium": 1, "low": 2}
    pending_clusters.sort(
        key=lambda item: (
            priority_order.get(item[1].get("priority", "low"), 2),
            -len(item[1].get("files", [])),
            item[1].get("type", ""),
            item[0],
        )
    )
    new_clusters = dict(pending_clusters)

    return {
        "version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "total_clusters": len(new_clusters),
            "by_priority": by_priority,
            "by_decision": by_decision,
        },
        "clusters": new_clusters,
    }


def yaml_representer_str(dumper: yaml.Dumper, data: str) -> yaml.Node:
    """Custom representer for strings to handle multiline properly."""
    if "\n" in data:
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)


def write_manifest(manifest: dict[str, Any]) -> None:
    """Write manifest to YAML file."""
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)

    yaml.add_representer(str, yaml_representer_str)

    with open(MANIFEST_PATH, "w") as f:
        f.write("# Test Deduplication Candidates Manifest\n")
        f.write("# Generated by: uv run agent-dedupe\n")
        f.write("# \n")
        f.write("# Cluster statuses: pending | in_progress | done\n")
        f.write("# Decisions: delete | merge | modernize\n")
        f.write("# Priorities: high | medium | low\n")
        f.write("#\n")
        yaml.dump(manifest, f, default_flow_style=False, sort_keys=False, allow_unicode=True)


def verify_manifest() -> int:
    """Verify manifest exists and is valid."""
    if not MANIFEST_PATH.exists():
        print(f"Error: Manifest not found: {MANIFEST_PATH}", file=sys.stderr)
        print("Run: uv run agent-dedupe  # to generate", file=sys.stderr)
        return 1

    try:
        with open(MANIFEST_PATH) as f:
            manifest = yaml.safe_load(f)

        if not manifest:
            print("Error: Manifest is empty", file=sys.stderr)
            return 1

        version = manifest.get("version")
        if version != 1:
            print(f"Error: Invalid manifest version: {version}", file=sys.stderr)
            return 1

        clusters = manifest.get("clusters", {})
        problems = []

        for cluster_id, cluster in clusters.items():
            # Check for stale in_progress without PR URL
            if cluster.get("status") == "in_progress" and not cluster.get("pr_url"):
                added = cluster.get("added")
                if added:
                    added_date = datetime.strptime(added, "%Y-%m-%d")
                    days_old = (datetime.now() - added_date).days
                    if days_old > 14:
                        problems.append(
                            f"Cluster {cluster_id}: in_progress for {days_old} days without pr_url"
                        )

            # Check for missing expected deltas
            if "expected_file_delta" not in cluster:
                problems.append(f"Cluster {cluster_id}: missing expected_file_delta")

        if problems:
            print("Manifest validation issues:", file=sys.stderr)
            for problem in problems:
                print(f"  - {problem}", file=sys.stderr)
            return 1

        print(f"Manifest valid: {len(clusters)} clusters")
        return 0

    except yaml.YAMLError as e:
        print(f"Error parsing manifest: {e}", file=sys.stderr)
        return 1


def show_cluster(cluster_id: str) -> int:
    """Show details for a specific cluster."""
    if not MANIFEST_PATH.exists():
        print(f"Error: Manifest not found: {MANIFEST_PATH}", file=sys.stderr)
        return 1

    with open(MANIFEST_PATH) as f:
        manifest = yaml.safe_load(f)

    clusters = manifest.get("clusters", {})

    # Support partial ID matching
    matches = [cid for cid in clusters if cid.startswith(cluster_id)]

    if not matches:
        print(f"Error: No cluster found matching '{cluster_id}'", file=sys.stderr)
        return 1

    if len(matches) > 1:
        print(f"Multiple clusters match '{cluster_id}':", file=sys.stderr)
        for match in matches:
            print(f"  - {match}", file=sys.stderr)
        return 1

    cluster = clusters[matches[0]]
    print(yaml.dump({matches[0]: cluster}, default_flow_style=False, sort_keys=False))
    return 0


def show_stats() -> int:
    """Show summary statistics."""
    if not MANIFEST_PATH.exists():
        print(f"Error: Manifest not found: {MANIFEST_PATH}", file=sys.stderr)
        return 1

    with open(MANIFEST_PATH) as f:
        manifest = yaml.safe_load(f)

    summary = manifest.get("summary", {})
    clusters = manifest.get("clusters", {})

    print("Test Deduplication Statistics")
    print("=" * 40)
    print(f"Total clusters: {summary.get('total_clusters', 0)}")
    print()
    print("By Priority:")
    for priority, count in summary.get("by_priority", {}).items():
        print(f"  {priority}: {count}")
    print()
    print("By Decision:")
    for decision, count in summary.get("by_decision", {}).items():
        print(f"  {decision}: {count}")
    print()
    print("By Status:")
    status_counts: dict[str, int] = defaultdict(int)
    for cluster in clusters.values():
        status_counts[cluster.get("status", "pending")] += 1
    for status, count in sorted(status_counts.items()):
        print(f"  {status}: {count}")
    print()

    # Calculate potential file reduction
    total_files = 0
    potential_reduction = 0
    for cluster in clusters.values():
        if cluster.get("status") != "done":
            total_files += len(cluster.get("files", []))
            potential_reduction += abs(cluster.get("expected_file_delta", 0))

    print(f"Files in pending clusters: {total_files}")
    print(f"Potential file reduction: -{potential_reduction}")

    return 0


def suggest_next_pr() -> int:
    """Suggest next PR packet (5-10 files, ordered by priority)."""
    if not MANIFEST_PATH.exists():
        print(f"Error: Manifest not found: {MANIFEST_PATH}", file=sys.stderr)
        return 1

    with open(MANIFEST_PATH) as f:
        manifest = yaml.safe_load(f)

    clusters = manifest.get("clusters", {})

    # Filter pending clusters and sort by priority
    priority_order = {"high": 0, "medium": 1, "low": 2}
    pending = [
        (cid, cluster) for cid, cluster in clusters.items() if cluster.get("status") == "pending"
    ]
    pending.sort(
        key=lambda x: (
            priority_order.get(x[1].get("priority", "low"), 2),
            -len(x[1].get("files", [])),
        )
    )

    if not pending:
        print("No pending clusters!")
        return 0

    print("Suggested Next PR Packet")
    print("=" * 40)

    total_files = 0
    selected = []

    for cluster_id, cluster in pending:
        file_count = len(cluster.get("files", []))
        if total_files + file_count <= 10 or not selected:
            selected.append((cluster_id, cluster))
            total_files += file_count
            if total_files >= 5:
                break

    for cluster_id, cluster in selected:
        print(f"\nCluster: {cluster_id}")
        print(f"  Type: {cluster.get('type')}")
        print(f"  Priority: {cluster.get('priority')}")
        print(f"  Decision: {cluster.get('decision')}")
        print(f"  Files ({len(cluster.get('files', []))}):")
        for f in cluster.get("files", [])[:5]:
            print(f"    - {f}")
        if len(cluster.get("files", [])) > 5:
            print(f"    ... and {len(cluster.get('files', [])) - 5} more")
        print(f"  Target: {cluster.get('target_location')}")

    print(f"\nTotal files in packet: {total_files}")
    print("\nTo start work:")
    print("  1. Update manifest: set status to 'in_progress', add owner")
    print("  2. Execute consolidation")
    print("  3. Run: uv run pytest -q <affected_files>")
    print("  4. Update manifest: set status to 'done', add pr_url")

    return 0


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate test deduplication candidates manifest")
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify manifest exists and is valid (for CI)",
    )
    parser.add_argument(
        "--cluster",
        type=str,
        metavar="ID",
        help="Show details for a specific cluster",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show summary statistics",
    )
    parser.add_argument(
        "--next-pr",
        action="store_true",
        help="Suggest next PR packet (5-10 files)",
    )
    parser.add_argument(
        "--source-fanout-threshold",
        type=int,
        default=SOURCE_FANOUT_THRESHOLD,
        help=f"Minimum test files for source fanout detection (default: {SOURCE_FANOUT_THRESHOLD})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print manifest to stdout instead of writing to file",
    )

    args = parser.parse_args()

    # Handle query modes
    if args.verify:
        return verify_manifest()

    if args.cluster:
        return show_cluster(args.cluster)

    if args.stats:
        return show_stats()

    if args.next_pr:
        return suggest_next_pr()

    # Generate mode
    print("Loading source test map...")
    source_test_map = load_source_test_map()

    print("Detecting source fanout clusters...")
    source_fanout_clusters = detect_source_fanout(
        source_test_map, threshold=args.source_fanout_threshold
    )
    print(f"  Found {len(source_fanout_clusters)} source fanout clusters")

    print("Detecting similar name clusters...")
    similar_name_clusters = detect_similar_names(source_test_map)
    print(f"  Found {len(similar_name_clusters)} similar name clusters")

    all_clusters = source_fanout_clusters + similar_name_clusters

    # Remove exact duplicate clusters deterministically (same file set + type + source modules).
    unique_clusters: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, tuple[str, ...], tuple[str, ...]]] = set()
    for cluster in sorted(all_clusters, key=lambda c: (c.get("type", ""), c.get("id", ""))):
        key = (
            cluster.get("type", ""),
            tuple(sorted(cluster.get("files", []))),
            tuple(sorted(cluster.get("source_modules", []))),
        )
        if key in seen_keys:
            continue
        seen_keys.add(key)
        unique_clusters.append(cluster)

    print(f"  {len(unique_clusters)} unique clusters after deduplication")

    print("Loading existing manifest for merge...")
    existing_manifest = load_existing_manifest()

    print("Building manifest...")
    manifest = build_manifest(unique_clusters, existing_manifest)

    if args.dry_run:
        print("\n--- DRY RUN OUTPUT ---")
        print(yaml.dump(manifest, default_flow_style=False, sort_keys=False))
        return 0

    print(f"Writing manifest to {MANIFEST_PATH}...")
    write_manifest(manifest)

    summary = manifest.get("summary", {})
    print(f"\nGenerated manifest with {summary.get('total_clusters', 0)} clusters")
    print(f"  High priority: {summary.get('by_priority', {}).get('high', 0)}")
    print(f"  Medium priority: {summary.get('by_priority', {}).get('medium', 0)}")
    print(f"  Low priority: {summary.get('by_priority', {}).get('low', 0)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
