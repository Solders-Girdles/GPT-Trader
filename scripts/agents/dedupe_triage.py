#!/usr/bin/env python3
"""Triage helper for test deduplication clusters.

This tool keeps reviewer decisions (accept/reject/defer) in a small, stable
triage file so we don't have to constantly rewrite the generated candidates
manifest.

Files:
  - Generated clusters: tests/_triage/dedupe_candidates.yaml
  - Triage decisions:  tests/_triage/dedupe_triage.yaml

Usage:
  uv run python scripts/agents/dedupe_triage.py --list
  uv run python scripts/agents/dedupe_triage.py --show 678535a285b3
  uv run python scripts/agents/dedupe_triage.py --set 72948f9bf0a3 rejected --owner rj --reason "Fanout cluster; not redundant."
  uv run python scripts/agents/dedupe_triage.py --clear 72948f9bf0a3
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).parent.parent.parent
CLUSTERS_PATH = PROJECT_ROOT / "tests" / "_triage" / "dedupe_candidates.yaml"
TRIAGE_PATH = PROJECT_ROOT / "tests" / "_triage" / "dedupe_triage.yaml"

VALID_TRIAGE_STATES = {"accepted", "rejected", "deferred"}


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def load_clusters_manifest() -> dict[str, Any]:
    manifest = _load_yaml(CLUSTERS_PATH)
    if not manifest:
        raise ValueError(f"Clusters manifest not found: {CLUSTERS_PATH}")
    if manifest.get("version") != 1:
        raise ValueError(f"Unsupported clusters manifest version: {manifest.get('version')!r}")
    clusters = manifest.get("clusters")
    if not isinstance(clusters, dict):
        raise ValueError("Clusters manifest missing/invalid 'clusters' mapping")
    return manifest


def load_triage() -> dict[str, Any]:
    triage = _load_yaml(TRIAGE_PATH)
    if not triage:
        return {"version": 1, "generated_at": None, "clusters": {}}
    if triage.get("version") != 1:
        raise ValueError(f"Unsupported triage version: {triage.get('version')!r}")
    clusters = triage.get("clusters")
    if clusters is None:
        triage["clusters"] = {}
    elif not isinstance(clusters, dict):
        raise ValueError("Triage file has invalid 'clusters' (expected mapping)")
    return triage


def resolve_cluster_id(cluster_id: str, clusters: dict[str, Any]) -> str:
    matches = [cid for cid in clusters if cid.startswith(cluster_id)]
    if not matches:
        raise ValueError(f"No cluster found matching '{cluster_id}'")
    if len(matches) > 1:
        raise ValueError(f"Multiple clusters match '{cluster_id}': {', '.join(matches[:5])}")
    return matches[0]


def write_triage(triage: dict[str, Any]) -> None:
    TRIAGE_PATH.parent.mkdir(parents=True, exist_ok=True)
    triage["generated_at"] = datetime.now(timezone.utc).isoformat()

    with open(TRIAGE_PATH, "w") as f:
        f.write("# Test Deduplication Triage\n")
        f.write("#\n")
        f.write("# States:\n")
        f.write("#   accepted: worth doing (keep surfaced)\n")
        f.write("#   rejected: false positive / not worth doing\n")
        f.write("#   deferred: maybe later (hide from suggestions)\n")
        f.write("#\n")
        yaml.dump(triage, f, default_flow_style=False, sort_keys=False, allow_unicode=True)


def list_clusters(
    *,
    manifest: dict[str, Any],
    triage: dict[str, Any],
    status: str,
    priority: str,
    cluster_type: str,
    triage_state: str,
) -> int:
    clusters: dict[str, Any] = manifest["clusters"]
    triaged: dict[str, Any] = triage.get("clusters", {})

    def matches_filters(cid: str, cluster: dict[str, Any]) -> bool:
        if status != "all" and cluster.get("status", "pending") != status:
            return False
        if priority != "all" and cluster.get("priority") != priority:
            return False
        if cluster_type != "all" and cluster.get("type") != cluster_type:
            return False
        state = triaged.get(cid, {}).get("state") or "untriaged"
        if triage_state != "all" and state != triage_state:
            return False
        return True

    rows: list[tuple[str, str, str, str, int, str]] = []
    for cid, cluster in clusters.items():
        if not isinstance(cluster, dict):
            continue
        if not matches_filters(cid, cluster):
            continue
        state = triaged.get(cid, {}).get("state") or "untriaged"
        rows.append(
            (
                cid,
                cluster.get("priority", "?"),
                cluster.get("status", "pending"),
                state,
                len(cluster.get("files", []) or []),
                cluster.get("target_location", ""),
            )
        )

    if not rows:
        print("No clusters match filters.")
        return 0

    print("cluster_id     priority  status      triage      files  target")
    print("-" * 80)
    for cid, prio, st, tri, files, target in rows:
        print(f"{cid:12}  {prio:8}  {st:10}  {tri:10}  {files:5}  {target}")
    return 0


def show_cluster(*, manifest: dict[str, Any], triage: dict[str, Any], cluster_id: str) -> int:
    clusters: dict[str, Any] = manifest["clusters"]
    resolved = resolve_cluster_id(cluster_id, clusters)
    cluster = clusters[resolved]
    state = (triage.get("clusters", {}).get(resolved, {}) or {}).get("state") or "untriaged"
    print(
        yaml.dump(
            {
                resolved: {
                    "cluster": cluster,
                    "triage": triage.get("clusters", {}).get(resolved),
                    "triage_state": state,
                }
            },
            sort_keys=False,
        )
    )
    return 0


def set_triage(
    *,
    manifest: dict[str, Any],
    triage: dict[str, Any],
    cluster_id: str,
    state: str,
    owner: str,
    reason: str | None,
) -> int:
    clusters: dict[str, Any] = manifest["clusters"]
    resolved = resolve_cluster_id(cluster_id, clusters)
    triaged: dict[str, Any] = triage.setdefault("clusters", {})

    if state not in VALID_TRIAGE_STATES:
        raise ValueError(f"Invalid triage state: {state!r} (valid: {sorted(VALID_TRIAGE_STATES)})")

    entry: dict[str, Any] = {
        "state": state,
        "owner": owner,
        "updated": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
    }
    if reason:
        entry["reason"] = reason
    triaged[resolved] = entry
    write_triage(triage)
    print(f"Updated triage: {resolved} -> {state}")
    return 0


def clear_triage(*, manifest: dict[str, Any], triage: dict[str, Any], cluster_id: str) -> int:
    clusters: dict[str, Any] = manifest["clusters"]
    resolved = resolve_cluster_id(cluster_id, clusters)
    triaged: dict[str, Any] = triage.get("clusters", {})
    if resolved not in triaged:
        print(f"No triage entry to clear for {resolved}")
        return 0
    del triaged[resolved]
    write_triage(triage)
    print(f"Cleared triage: {resolved}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Triage helper for dedupe clusters.")
    parser.add_argument("--list", action="store_true", help="List clusters with triage state")
    parser.add_argument("--show", type=str, metavar="ID", help="Show one cluster + triage details")
    parser.add_argument(
        "--set",
        nargs=2,
        metavar=("ID", "STATE"),
        help="Set triage state for a cluster (accepted|rejected|deferred)",
    )
    parser.add_argument("--clear", type=str, metavar="ID", help="Remove triage entry for a cluster")
    parser.add_argument("--owner", type=str, help="Owner when using --set")
    parser.add_argument("--reason", type=str, help="Reason when using --set")
    parser.add_argument(
        "--status", type=str, default="pending", choices=["pending", "in_progress", "done", "all"]
    )
    parser.add_argument(
        "--priority", type=str, default="all", choices=["high", "medium", "low", "all"]
    )
    parser.add_argument(
        "--type",
        dest="cluster_type",
        type=str,
        default="all",
        choices=["source_fanout", "similar_names", "fixture_repeat", "all"],
    )
    parser.add_argument(
        "--triage-state",
        type=str,
        default="all",
        choices=["accepted", "rejected", "deferred", "untriaged", "all"],
        help="Filter by triage state (default: all)",
    )

    args = parser.parse_args()

    try:
        manifest = load_clusters_manifest()
        triage = load_triage()
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    actions = sum(bool(x) for x in [args.list, args.show, args.set, args.clear])
    if actions != 1:
        print("Error: choose exactly one of --list, --show, --set, --clear", file=sys.stderr)
        return 1

    try:
        if args.list:
            return list_clusters(
                manifest=manifest,
                triage=triage,
                status=args.status,
                priority=args.priority,
                cluster_type=args.cluster_type,
                triage_state=args.triage_state,
            )
        if args.show:
            return show_cluster(manifest=manifest, triage=triage, cluster_id=args.show)
        if args.set:
            cluster_id, state = args.set
            if not args.owner:
                print("Error: --owner is required with --set", file=sys.stderr)
                return 1
            return set_triage(
                manifest=manifest,
                triage=triage,
                cluster_id=cluster_id,
                state=state,
                owner=args.owner,
                reason=args.reason,
            )
        if args.clear:
            return clear_triage(manifest=manifest, triage=triage, cluster_id=args.clear)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
