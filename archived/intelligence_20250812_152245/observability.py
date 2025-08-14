from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any


class ObservabilityFramework:
    """Structured logging, tracing, and evaluation harness."""

    def __init__(self, log_dir: Path) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger("intelligence")
        if not self.logger.handlers:
            handler = logging.FileHandler(self.log_dir / "intelligence.log")
            handler.setFormatter(
                logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            )
            self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    def log_decision(
        self, decision_type: str, decision_data: dict[str, Any], metadata: dict[str, Any]
    ) -> None:
        entry = {
            "timestamp": datetime.now().isoformat(),
            "decision_type": decision_type,
            "decision_data": decision_data,
            "metadata": metadata,
        }
        self.logger.info(json.dumps(entry))

    def log_metrics(self, metrics: dict[str, float], model_version: str) -> None:
        entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": "metrics",
            "model_version": model_version,
            "metrics": metrics,
        }
        self.logger.info(json.dumps(entry))

    def create_evaluation_harness(self, test_window: str = "30d") -> EvaluationHarness:
        return EvaluationHarness(self.log_dir, test_window)


class EvaluationHarness:
    """Lightweight evaluation harness for model comparison."""

    def __init__(self, log_dir: Path, test_window: str) -> None:
        self.log_dir = Path(log_dir)
        self.test_window = test_window

    def replay_fixed_window(self, model_versions: list[str], test_data) -> dict[str, dict[str, float]]:  # type: ignore[no-untyped-def]
        results: dict[str, dict[str, float]] = {}
        for version in model_versions:
            predictions = self.run_inference(self.load_model_artifacts(version), test_data)
            metrics = self.calculate_metrics(predictions, test_data)
            results[version] = metrics
        return results

    # The following are placeholders to enable wiring later without import churn.
    def load_model_artifacts(self, version: str):  # pragma: no cover - placeholder
        return {"version": version}

    def run_inference(self, artifacts, test_data):  # pragma: no cover - placeholder
        return []

    def calculate_metrics(
        self, predictions, test_data
    ) -> dict[str, float]:  # pragma: no cover - placeholder
        return {"sharpe_ratio": 0.0}
