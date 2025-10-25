"""Monitoring and analytics logging mixin."""

from __future__ import annotations

from typing import Any

from .levels import LogLevel


class MonitoringLoggingMixin:
    """Provide monitoring and analytics logging helpers."""

    def log_ml_prediction(
        self,
        model_name: str,
        prediction: Any,
        confidence: float | None = None,
        input_features: dict[str, Any] | None = None,
        inference_time_ms: float | None = None,
        **kwargs: Any,
    ) -> None:
        entry = self._create_log_entry(
            level=LogLevel.INFO,
            event_type="ml_prediction",
            message=f"Model {model_name} predicted: {prediction}",
            model_name=model_name,
            prediction=str(prediction),
            **kwargs,
        )
        if confidence is not None:
            entry["confidence"] = confidence
        if input_features:
            entry["feature_count"] = len(input_features)
            entry["sample_features"] = {k: v for k, v in list(input_features.items())[:5]}
        if inference_time_ms is not None:
            entry["inference_time_ms"] = inference_time_ms
        self._emit_log(entry)

    def log_performance(
        self,
        operation: str,
        duration_ms: float,
        success: bool = True,
        **kwargs: Any,
    ) -> None:
        level = LogLevel.INFO if success else LogLevel.WARNING
        entry = self._create_log_entry(
            level=level,
            event_type="performance_metric",
            message=f"{operation} completed in {duration_ms:.2f}ms",
            operation=operation,
            duration_ms=duration_ms,
            success=success,
            **kwargs,
        )
        self._emit_log(entry)

    def log_strategy_duration(self, strategy: str, duration_ms: float, **kwargs: Any) -> None:
        entry = self._create_log_entry(
            level=LogLevel.INFO,
            event_type="strategy_duration",
            message=f"{strategy} took {duration_ms:.1f}ms",
            strategy=strategy,
            duration_ms=duration_ms,
            **kwargs,
        )
        self._emit_log(entry)

    def log_risk_breach(
        self,
        limit_type: str,
        limit_value: float,
        current_value: float,
        **kwargs: Any,
    ) -> None:
        entry = self._create_log_entry(
            level=LogLevel.WARNING,
            event_type="risk_limit_breach",
            message=f"risk breach {limit_type}",
            limit_type=limit_type,
            limit_value=limit_value,
            current_value=current_value,
            exceeded_by=current_value - limit_value,
            **kwargs,
        )
        self._emit_log(entry)


__all__ = ["MonitoringLoggingMixin"]
