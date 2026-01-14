from __future__ import annotations

from scripts.agents import generate_reasoning_artifacts


def test_guard_stack_map_clusters() -> None:
    guard_map = generate_reasoning_artifacts.build_guard_stack_map()

    cluster_ids = {cluster["id"] for cluster in guard_map["clusters"]}
    assert "preflight" in cluster_ids
    assert "runtime" in cluster_ids

    node_clusters = {node["cluster"] for node in guard_map["nodes"]}
    assert "preflight" in node_clusters
    assert "runtime" in node_clusters


def test_guard_stack_dot_includes_clusters() -> None:
    guard_map = generate_reasoning_artifacts.build_guard_stack_map()
    dot = "\n".join(generate_reasoning_artifacts.build_guard_stack_dot(guard_map))

    assert "cluster_preflight" in dot
    assert "cluster_runtime" in dot
