from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


class MetricsRegistry:
    """Central place to log and compare metrics across versions."""

    def __init__(self, registry_path: Path) -> None:
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)

    def log_metrics(
        self, model_version: str, metrics: dict[str, float], metadata: dict[str, Any]
    ) -> None:
        entry = {
            "model_version": model_version,
            "metrics": metrics,
            "metadata": metadata,
            "timestamp": datetime.now().isoformat(),
        }
        metrics_file = self.registry_path / f"metrics_v{model_version}.json"
        with open(metrics_file, "w") as f:
            json.dump(entry, f, indent=2)

    def get_metrics(self, model_version: str) -> dict[str, Any]:
        metrics_file = self.registry_path / f"metrics_v{model_version}.json"
        if not metrics_file.exists():
            raise FileNotFoundError(f"Metrics for version {model_version} not found")
        with open(metrics_file) as f:
            return json.load(f)

    def compare_versions(self, versions: list[str]) -> dict[str, dict[str, float]]:
        comparison: dict[str, dict[str, float]] = {}
        for version in versions:
            metrics_data = self.get_metrics(version)
            comparison[version] = metrics_data.get("metrics", {})
        return comparison

    def list_versions(self) -> list[str]:
        versions: list[str] = []
        for path in self.registry_path.glob("metrics_v*.json"):
            stem = path.stem  # e.g., metrics_vportfolio_20250101T120000
            version = stem.replace("metrics_v", "")
            versions.append(version)
        versions.sort()
        return versions
