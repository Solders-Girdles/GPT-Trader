"""
Result analysis for optimization outcomes.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


def _convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: _convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [_convert_numpy_types(item) for item in obj]
    else:
        return obj


class ResultAnalyzer:
    """Analyzes optimization results and provides insights."""

    def __init__(self) -> None:
        pass

    def analyze_results(self, results: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Perform comprehensive analysis of optimization results.

        Args:
            results: List of result dictionaries from optimization

        Returns:
            Dictionary containing analysis results
        """
        if not results:
            return {"error": "No results to analyze"}

        # Convert to DataFrame for easier analysis
        df = self._results_to_dataframe(results)

        analysis = {
            "summary_statistics": self._calculate_summary_statistics(df),
            "parameter_analysis": self._analyze_parameters(df),
            "correlation_analysis": self._analyze_correlations(df),
            "top_performers": self._find_top_performers(df),
            "robustness_analysis": self._analyze_robustness(df),
            "sensitivity_analysis": self._analyze_sensitivity(df),
        }

        return analysis

    def _results_to_dataframe(self, results: list[dict[str, Any]]) -> pd.DataFrame:
        """Convert results to pandas DataFrame."""
        # Flatten results
        flattened = []
        for result in results:
            row = {}

            # Add metrics
            for key in ["sharpe", "cagr", "max_drawdown", "total_return", "n_trades"]:
                if key in result:
                    row[key] = result[key]

            # Add parameters
            if "params" in result:
                for param_name, param_value in result["params"].items():
                    row[f"param_{param_name}"] = param_value

            # Add other fields
            for key, value in result.items():
                if key not in [
                    "params",
                    "sharpe",
                    "cagr",
                    "max_drawdown",
                    "total_return",
                    "n_trades",
                ]:
                    row[key] = value

            flattened.append(row)

        return pd.DataFrame(flattened)

    def _calculate_summary_statistics(self, df: pd.DataFrame) -> dict[str, Any]:
        """Calculate summary statistics for all metrics."""
        metrics = ["sharpe", "cagr", "max_drawdown", "total_return", "n_trades"]
        summary = {}

        for metric in metrics:
            if metric in df.columns:
                values = df[metric].dropna()
                if len(values) > 0:
                    summary[metric] = {
                        "count": len(values),
                        "mean": float(values.mean()),
                        "std": float(values.std()),
                        "min": float(values.min()),
                        "max": float(values.max()),
                        "median": float(values.median()),
                        "q25": float(values.quantile(0.25)),
                        "q75": float(values.quantile(0.75)),
                    }

        return summary

    def _analyze_parameters(self, df: pd.DataFrame) -> dict[str, Any]:
        """Analyze parameter distributions and relationships."""
        param_cols = [col for col in df.columns if col.startswith("param_")]
        analysis = {}

        for param_col in param_cols:
            param_name = param_col.replace("param_", "")
            values = df[param_col].dropna()

            if len(values) > 0:
                analysis[param_name] = {
                    "unique_values": int(values.nunique()),
                    "most_common": values.mode().iloc[0] if len(values.mode()) > 0 else None,
                    "distribution": {
                        "min": float(values.min()),
                        "max": float(values.max()),
                        "mean": float(values.mean()),
                        "std": float(values.std()),
                    },
                }

        return analysis

    def _analyze_correlations(self, df: pd.DataFrame) -> dict[str, Any]:
        """Analyze correlations between parameters and performance metrics."""
        param_cols = [col for col in df.columns if col.startswith("param_")]
        metric_cols = ["sharpe", "cagr", "max_drawdown", "total_return"]

        # Filter to columns that exist
        param_cols = [col for col in param_cols if col in df.columns]
        metric_cols = [col for col in metric_cols if col in df.columns]

        correlations = {}

        # Parameter-parameter correlations
        if len(param_cols) > 1:
            param_corr = df[param_cols].corr()
            correlations["parameter_correlations"] = param_corr.to_dict()

        # Parameter-metric correlations
        if param_cols and metric_cols:
            param_metric_corr = df[param_cols + metric_cols].corr()
            # Extract only parameter-metric correlations
            param_metric_subset = param_metric_corr.loc[param_cols, metric_cols]
            correlations["parameter_metric_correlations"] = param_metric_subset.to_dict()

        return correlations

    def _find_top_performers(self, df: pd.DataFrame, top_n: int = 10) -> dict[str, Any]:
        """Find top performing parameter combinations."""
        if "sharpe" not in df.columns:
            return {"error": "Sharpe ratio not found in results"}

        # Sort by Sharpe ratio
        top_sharpe = df.nlargest(top_n, "sharpe")

        # Sort by CAGR
        top_cagr = df.nlargest(top_n, "cagr") if "cagr" in df.columns else None

        # Sort by risk-adjusted return (Sharpe / MaxDD)
        if "max_drawdown" in df.columns:
            df_copy = df.copy()
            df_copy["risk_adjusted"] = df_copy["sharpe"] / (df_copy["max_drawdown"] + 1e-6)
            top_risk_adj = df_copy.nlargest(top_n, "risk_adjusted")
        else:
            top_risk_adj = None

        return {
            "top_by_sharpe": top_sharpe.to_dict("records"),
            "top_by_cagr": top_cagr.to_dict("records") if top_cagr is not None else None,
            "top_by_risk_adjusted": (
                top_risk_adj.to_dict("records") if top_risk_adj is not None else None
            ),
        }

    def _analyze_robustness(self, df: pd.DataFrame) -> dict[str, Any]:
        """Analyze robustness of parameter combinations."""
        if "sharpe" not in df.columns:
            return {"error": "Sharpe ratio not found in results"}

        # Find parameter combinations that appear multiple times
        param_cols = [col for col in df.columns if col.startswith("param_")]

        if param_cols:
            # Group by parameters and analyze performance consistency
            grouped = df.groupby(param_cols)["sharpe"].agg(["count", "mean", "std"])
            grouped = grouped.reset_index()

            # Find robust combinations (low std, high mean)
            robust = grouped[
                (grouped["count"] >= 2)  # Appears multiple times
                & (grouped["std"] < grouped["std"].quantile(0.25))  # Low variance
                & (grouped["mean"] > grouped["mean"].quantile(0.75))  # High performance
            ]

            return {
                "robust_combinations": robust.to_dict("records"),
                "consistency_analysis": {
                    "high_consistency": len(
                        grouped[grouped["std"] < grouped["std"].quantile(0.25)]
                    ),
                    "low_consistency": len(grouped[grouped["std"] > grouped["std"].quantile(0.75)]),
                },
            }

        return {"error": "No parameter columns found"}

    def _analyze_sensitivity(self, df: pd.DataFrame) -> dict[str, Any]:
        """Analyze parameter sensitivity to performance."""
        param_cols = [col for col in df.columns if col.startswith("param_")]
        metrics = ["sharpe", "cagr", "max_drawdown", "total_return"]

        sensitivity = {}

        for metric in metrics:
            if metric not in df.columns:
                continue

            metric_sensitivity = {}
            for param_col in param_cols:
                param_name = param_col.replace("param_", "")

                # Calculate correlation
                correlation = df[param_col].corr(df[metric])

                # Calculate sensitivity using regression
                try:
                    slope, intercept, r_value, p_value, std_err = stats.linregress(
                        df[param_col].dropna(), df[metric].dropna()
                    )

                    metric_sensitivity[param_name] = {
                        "correlation": float(correlation),
                        "slope": float(slope),
                        "r_squared": float(r_value**2),
                        "p_value": float(p_value),
                        "sensitivity": (
                            "high"
                            if abs(correlation) > 0.3
                            else "medium" if abs(correlation) > 0.1 else "low"
                        ),
                    }
                except (ValueError, np.linalg.LinAlgError, TypeError) as e:
                    # Linear regression failed due to insufficient data or numerical issues
                    logger.debug(f"Linear regression failed for parameter {param_name}: {e}")
                    metric_sensitivity[param_name] = {
                        "correlation": float(correlation),
                        "sensitivity": "unknown",
                    }

            sensitivity[metric] = metric_sensitivity

        return sensitivity

    def generate_report(self, results: list[dict[str, Any]], output_path: str) -> None:
        """Generate a comprehensive analysis report."""
        analysis = self.analyze_results(results)

        # Create formatted report
        report = {
            "analysis": analysis,
            "recommendations": self._generate_recommendations(analysis),
            "timestamp": pd.Timestamp.now().isoformat(),
        }

        # Convert numpy types and save report
        report = _convert_numpy_types(report)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"Analysis report saved to {output_path}")

    def _generate_recommendations(self, analysis: dict[str, Any]) -> list[str]:
        """Generate recommendations based on analysis."""
        recommendations = []

        # Check for high correlations
        if "correlation_analysis" in analysis:
            corr_analysis = analysis["correlation_analysis"]
            if "parameter_correlations" in corr_analysis:
                param_corr = corr_analysis["parameter_correlations"]
                high_corr_pairs = []

                for param1, correlations in param_corr.items():
                    for param2, corr_value in correlations.items():
                        if param1 != param2 and abs(corr_value) > 0.8:
                            high_corr_pairs.append((param1, param2, corr_value))

                if high_corr_pairs:
                    recommendations.append(
                        f"High parameter correlations detected: {high_corr_pairs[:3]}. "
                        "Consider removing redundant parameters."
                    )

        # Check for parameter sensitivity
        if "sensitivity_analysis" in analysis:
            sens_analysis = analysis["sensitivity_analysis"]
            if "sharpe" in sens_analysis:
                high_sens_params = [
                    param
                    for param, sens in sens_analysis["sharpe"].items()
                    if sens.get("sensitivity") == "high"
                ]
                if high_sens_params:
                    recommendations.append(
                        f"High sensitivity parameters for Sharpe ratio: {high_sens_params}. "
                        "Focus optimization efforts on these parameters."
                    )

        # Check for robust combinations
        if "robustness_analysis" in analysis:
            robust_analysis = analysis["robustness_analysis"]
            if "robust_combinations" in robust_analysis:
                robust_count = len(robust_analysis["robust_combinations"])
                if robust_count > 0:
                    recommendations.append(
                        f"Found {robust_count} robust parameter combinations. "
                        "These are preferred for live trading."
                    )

        return recommendations
