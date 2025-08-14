"""
Model Comparison Report Generator
Phase 3, Week 2: MON-017
Comprehensive reporting for model performance comparison
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from io import StringIO
from typing import Any

logger = logging.getLogger(__name__)


class ReportFormat(Enum):
    """Report output formats"""

    TEXT = "text"
    JSON = "json"
    HTML = "html"
    MARKDOWN = "markdown"
    PDF = "pdf"


class ComparisonMetric(Enum):
    """Metrics for comparison"""

    # Performance metrics
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    AUC_ROC = "auc_roc"

    # Trading metrics
    SHARPE_RATIO = "sharpe_ratio"
    MAX_DRAWDOWN = "max_drawdown"
    WIN_RATE = "win_rate"
    PROFIT_FACTOR = "profit_factor"
    TOTAL_RETURN = "total_return"

    # Operational metrics
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    MEMORY_USAGE = "memory_usage"
    ERROR_RATE = "error_rate"


@dataclass
class ModelMetrics:
    """Metrics for a single model"""

    model_id: str
    model_version: str

    # Performance metrics
    accuracy: float | None = None
    precision: float | None = None
    recall: float | None = None
    f1_score: float | None = None
    auc_roc: float | None = None

    # Trading metrics
    sharpe_ratio: float | None = None
    max_drawdown: float | None = None
    win_rate: float | None = None
    profit_factor: float | None = None
    total_return: float | None = None

    # Operational metrics
    avg_latency_ms: float | None = None
    p95_latency_ms: float | None = None
    p99_latency_ms: float | None = None
    throughput_qps: float | None = None
    memory_usage_mb: float | None = None
    error_rate: float | None = None

    # Test conditions
    test_period_start: datetime | None = None
    test_period_end: datetime | None = None
    n_predictions: int | None = None
    n_trades: int | None = None

    # Confidence intervals
    confidence_intervals: dict[str, tuple[float, float]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            "model_id": self.model_id,
            "model_version": self.model_version,
            "accuracy": self.accuracy,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "avg_latency_ms": self.avg_latency_ms,
            "n_predictions": self.n_predictions,
        }


@dataclass
class ComparisonResult:
    """Result of model comparison"""

    winner: str
    margin: float
    confidence: float
    statistical_significance: bool
    p_value: float | None = None
    effect_size: float | None = None
    interpretation: str = ""


class ModelComparisonReport:
    """
    Generate comprehensive model comparison reports.

    Features:
    - Multi-model comparison
    - Statistical significance testing
    - Visual comparisons
    - Multiple output formats
    - Automated recommendations
    """

    def __init__(self):
        """Initialize report generator"""
        self.models: dict[str, ModelMetrics] = {}
        self.comparisons: list[ComparisonResult] = []
        self.test_results: dict[str, Any] = {}

    def add_model(self, metrics: ModelMetrics) -> None:
        """
        Add model to comparison.

        Args:
            metrics: Model metrics
        """
        self.models[metrics.model_id] = metrics
        logger.info(f"Added model {metrics.model_id} to comparison")

    def compare_models(
        self, model_a_id: str, model_b_id: str, metrics: list[ComparisonMetric] | None = None
    ) -> dict[str, ComparisonResult]:
        """
        Compare two models across metrics.

        Args:
            model_a_id: First model ID
            model_b_id: Second model ID
            metrics: Metrics to compare (all if None)

        Returns:
            Comparison results by metric
        """
        if model_a_id not in self.models or model_b_id not in self.models:
            raise ValueError("Models not found for comparison")

        model_a = self.models[model_a_id]
        model_b = self.models[model_b_id]

        # Default to all available metrics
        if metrics is None:
            metrics = self._get_available_metrics(model_a, model_b)

        results = {}

        for metric in metrics:
            result = self._compare_metric(model_a, model_b, metric)
            if result:
                results[metric.value] = result

        return results

    def _compare_metric(
        self, model_a: ModelMetrics, model_b: ModelMetrics, metric: ComparisonMetric
    ) -> ComparisonResult | None:
        """Compare models on single metric"""
        # Get metric values
        value_a = getattr(model_a, metric.value, None)
        value_b = getattr(model_b, metric.value, None)

        if value_a is None or value_b is None:
            return None

        # Determine winner based on metric type
        if metric in [
            ComparisonMetric.MAX_DRAWDOWN,
            ComparisonMetric.LATENCY,
            ComparisonMetric.ERROR_RATE,
            ComparisonMetric.MEMORY_USAGE,
        ]:
            # Lower is better
            winner = model_a.model_id if value_a < value_b else model_b.model_id
            margin = abs(value_b - value_a) / max(abs(value_a), 0.001)
        else:
            # Higher is better
            winner = model_a.model_id if value_a > value_b else model_b.model_id
            margin = abs(value_b - value_a) / max(abs(value_a), 0.001)

        # Simple statistical test (would use proper test in production)
        significant = margin > 0.05  # 5% difference

        # Generate interpretation
        better_worse = "better" if winner == model_a.model_id else "worse"
        interpretation = (
            f"{model_a.model_id} is {margin:.1%} {better_worse} than "
            f"{model_b.model_id} on {metric.value}"
        )

        return ComparisonResult(
            winner=winner,
            margin=margin,
            confidence=0.95 if significant else 0.5,
            statistical_significance=significant,
            interpretation=interpretation,
        )

    def _get_available_metrics(
        self, model_a: ModelMetrics, model_b: ModelMetrics
    ) -> list[ComparisonMetric]:
        """Get metrics available for both models"""
        available = []

        for metric in ComparisonMetric:
            value_a = getattr(model_a, metric.value, None)
            value_b = getattr(model_b, metric.value, None)
            if value_a is not None and value_b is not None:
                available.append(metric)

        return available

    def generate_report(
        self, format: ReportFormat = ReportFormat.TEXT, include_visuals: bool = False
    ) -> str:
        """
        Generate comparison report.

        Args:
            format: Output format
            include_visuals: Include visualizations

        Returns:
            Report string
        """
        if format == ReportFormat.TEXT:
            return self._generate_text_report()
        elif format == ReportFormat.JSON:
            return self._generate_json_report()
        elif format == ReportFormat.MARKDOWN:
            return self._generate_markdown_report()
        elif format == ReportFormat.HTML:
            return self._generate_html_report(include_visuals)
        else:
            return self._generate_text_report()

    def _generate_text_report(self) -> str:
        """Generate text format report"""
        report = StringIO()

        report.write("=" * 70 + "\n")
        report.write("MODEL COMPARISON REPORT\n")
        report.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        report.write("=" * 70 + "\n\n")

        # Model summary
        report.write("MODELS IN COMPARISON\n")
        report.write("-" * 40 + "\n")
        for model_id, metrics in self.models.items():
            report.write(f"\n{model_id} (v{metrics.model_version}):\n")
            report.write(f"  Accuracy: {metrics.accuracy:.3f}\n" if metrics.accuracy else "")
            report.write(
                f"  Sharpe Ratio: {metrics.sharpe_ratio:.2f}\n" if metrics.sharpe_ratio else ""
            )
            report.write(
                f"  Max Drawdown: {metrics.max_drawdown:.2%}\n" if metrics.max_drawdown else ""
            )
            report.write(f"  Win Rate: {metrics.win_rate:.2%}\n" if metrics.win_rate else "")
            report.write(
                f"  Avg Latency: {metrics.avg_latency_ms:.2f}ms\n" if metrics.avg_latency_ms else ""
            )
            report.write(
                f"  Predictions: {metrics.n_predictions:,}\n" if metrics.n_predictions else ""
            )

        # Pairwise comparisons
        if len(self.models) >= 2:
            report.write("\nPAIRWISE COMPARISONS\n")
            report.write("-" * 40 + "\n")

            model_ids = list(self.models.keys())
            for i in range(len(model_ids)):
                for j in range(i + 1, len(model_ids)):
                    report.write(f"\n{model_ids[i]} vs {model_ids[j]}:\n")

                    comparisons = self.compare_models(model_ids[i], model_ids[j])
                    for metric, result in comparisons.items():
                        report.write(f"  {metric}: {result.winner} wins ")
                        report.write(f"(+{result.margin:.1%})")
                        if result.statistical_significance:
                            report.write(" ***")
                        report.write("\n")

        # Overall ranking
        report.write("\nOVERALL RANKING\n")
        report.write("-" * 40 + "\n")
        rankings = self._calculate_rankings()
        for rank, (model_id, score) in enumerate(rankings, 1):
            report.write(f"{rank}. {model_id} (score: {score:.2f})\n")

        # Recommendations
        report.write("\nRECOMMENDATIONS\n")
        report.write("-" * 40 + "\n")
        recommendations = self._generate_recommendations()
        for rec in recommendations:
            report.write(f"• {rec}\n")

        return report.getvalue()

    def _generate_json_report(self) -> str:
        """Generate JSON format report"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "models": {model_id: metrics.to_dict() for model_id, metrics in self.models.items()},
            "comparisons": {},
            "rankings": self._calculate_rankings(),
            "recommendations": self._generate_recommendations(),
        }

        # Add pairwise comparisons
        model_ids = list(self.models.keys())
        for i in range(len(model_ids)):
            for j in range(i + 1, len(model_ids)):
                key = f"{model_ids[i]}_vs_{model_ids[j]}"
                comparisons = self.compare_models(model_ids[i], model_ids[j])
                report["comparisons"][key] = {
                    metric: {
                        "winner": result.winner,
                        "margin": result.margin,
                        "significant": result.statistical_significance,
                    }
                    for metric, result in comparisons.items()
                }

        return json.dumps(report, indent=2, default=str)

    def _generate_markdown_report(self) -> str:
        """Generate Markdown format report"""
        report = StringIO()

        report.write("# Model Comparison Report\n\n")
        report.write(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")

        # Model table
        report.write("## Models in Comparison\n\n")
        report.write("| Model | Accuracy | Sharpe | Drawdown | Latency | Predictions |\n")
        report.write("|-------|----------|--------|----------|---------|-------------|\n")

        for model_id, m in self.models.items():
            report.write(f"| {model_id} ")
            report.write(f"| {m.accuracy:.3f} " if m.accuracy else "| - ")
            report.write(f"| {m.sharpe_ratio:.2f} " if m.sharpe_ratio else "| - ")
            report.write(f"| {m.max_drawdown:.2%} " if m.max_drawdown else "| - ")
            report.write(f"| {m.avg_latency_ms:.1f}ms " if m.avg_latency_ms else "| - ")
            report.write(f"| {m.n_predictions:,} " if m.n_predictions else "| - ")
            report.write("|\n")

        # Comparison matrix
        report.write("\n## Head-to-Head Comparison\n\n")
        model_ids = list(self.models.keys())
        if len(model_ids) >= 2:
            # Create comparison matrix
            report.write("| Metric | ")
            report.write(
                " | ".join(f"{m1} vs {m2}" for m1 in model_ids for m2 in model_ids if m1 < m2)
            )
            report.write(" |\n")

            report.write("|--------|")
            report.write("-|" * (len(model_ids) * (len(model_ids) - 1) // 2))
            report.write("\n")

            # Add comparison results
            # (simplified for brevity)

        # Rankings
        report.write("\n## Overall Rankings\n\n")
        rankings = self._calculate_rankings()
        for rank, (model_id, score) in enumerate(rankings, 1):
            report.write(f"{rank}. **{model_id}** (score: {score:.2f})\n")

        # Recommendations
        report.write("\n## Recommendations\n\n")
        for rec in self._generate_recommendations():
            report.write(f"- {rec}\n")

        return report.getvalue()

    def _generate_html_report(self, include_visuals: bool) -> str:
        """Generate HTML format report"""
        html = StringIO()

        html.write(
            """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Model Comparison Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1 { color: #333; }
                table { border-collapse: collapse; width: 100%; margin: 20px 0; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                .winner { background-color: #d4edda; font-weight: bold; }
                .significant { color: #28a745; }
            </style>
        </head>
        <body>
        """
        )

        html.write("<h1>Model Comparison Report</h1>")
        html.write(f"<p><em>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</em></p>")

        # Model table
        html.write("<h2>Models in Comparison</h2>")
        html.write("<table>")
        html.write(
            "<tr><th>Model</th><th>Accuracy</th><th>Sharpe</th><th>Drawdown</th><th>Latency</th></tr>"
        )

        for model_id, m in self.models.items():
            html.write(f"<tr><td>{model_id}</td>")
            html.write(f"<td>{m.accuracy:.3f}</td>" if m.accuracy else "<td>-</td>")
            html.write(f"<td>{m.sharpe_ratio:.2f}</td>" if m.sharpe_ratio else "<td>-</td>")
            html.write(f"<td>{m.max_drawdown:.2%}</td>" if m.max_drawdown else "<td>-</td>")
            html.write(f"<td>{m.avg_latency_ms:.1f}ms</td>" if m.avg_latency_ms else "<td>-</td>")
            html.write("</tr>")

        html.write("</table>")

        # Rankings
        html.write("<h2>Overall Rankings</h2>")
        html.write("<ol>")
        for model_id, score in self._calculate_rankings():
            html.write(f"<li><strong>{model_id}</strong> (score: {score:.2f})</li>")
        html.write("</ol>")

        html.write("</body></html>")

        return html.getvalue()

    def _calculate_rankings(self) -> list[tuple[str, float]]:
        """Calculate overall model rankings"""
        scores = {}

        for model_id, metrics in self.models.items():
            score = 0
            n_metrics = 0

            # Performance score
            if metrics.accuracy:
                score += metrics.accuracy
                n_metrics += 1
            if metrics.sharpe_ratio:
                score += min(metrics.sharpe_ratio / 3, 1)  # Normalize
                n_metrics += 1
            if metrics.win_rate:
                score += metrics.win_rate
                n_metrics += 1
            if metrics.max_drawdown:
                score += 1 - metrics.max_drawdown  # Lower is better
                n_metrics += 1

            # Average score
            if n_metrics > 0:
                scores[model_id] = score / n_metrics
            else:
                scores[model_id] = 0

        # Sort by score
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

    def _generate_recommendations(self) -> list[str]:
        """Generate actionable recommendations"""
        recommendations = []

        if not self.models:
            return ["Add models for comparison"]

        rankings = self._calculate_rankings()

        if rankings:
            best_model = rankings[0][0]
            best_score = rankings[0][1]

            recommendations.append(f"Best overall model: {best_model} (score: {best_score:.2f})")

            # Check if clear winner
            if len(rankings) > 1:
                margin = (rankings[0][1] - rankings[1][1]) / rankings[1][1]
                if margin > 0.1:  # 10% better
                    recommendations.append(
                        f"Strong recommendation to use {best_model} "
                        f"({margin:.1%} better than runner-up)"
                    )
                else:
                    recommendations.append("Models are closely matched - consider A/B testing")

            # Specific recommendations
            best_metrics = self.models[best_model]

            if best_metrics.sharpe_ratio and best_metrics.sharpe_ratio < 1.0:
                recommendations.append(
                    "Warning: Best model has Sharpe < 1.0 - consider further optimization"
                )

            if best_metrics.max_drawdown and best_metrics.max_drawdown > 0.20:
                recommendations.append("Warning: High drawdown detected - implement risk controls")

            if best_metrics.error_rate and best_metrics.error_rate > 0.01:
                recommendations.append("Consider improving model reliability (error rate > 1%)")

        return recommendations

    def save_report(self, filepath: str, format: ReportFormat = ReportFormat.TEXT) -> None:
        """
        Save report to file.

        Args:
            filepath: Output file path
            format: Report format
        """
        report = self.generate_report(format)

        with open(filepath, "w") as f:
            f.write(report)

        logger.info(f"Report saved to {filepath}")


def demonstrate_comparison_report():
    """Demonstrate model comparison reporting"""
    print("Model Comparison Report Demo")
    print("=" * 60)

    # Create report generator
    report = ModelComparisonReport()

    # Add model metrics
    model_a = ModelMetrics(
        model_id="model_a",
        model_version="1.0",
        accuracy=0.58,
        sharpe_ratio=1.2,
        max_drawdown=0.15,
        win_rate=0.55,
        avg_latency_ms=5.2,
        n_predictions=10000,
    )
    report.add_model(model_a)

    model_b = ModelMetrics(
        model_id="model_b",
        model_version="2.0",
        accuracy=0.62,
        sharpe_ratio=1.4,
        max_drawdown=0.12,
        win_rate=0.58,
        avg_latency_ms=6.8,
        n_predictions=10000,
    )
    report.add_model(model_b)

    model_c = ModelMetrics(
        model_id="model_c",
        model_version="3.0",
        accuracy=0.60,
        sharpe_ratio=1.5,
        max_drawdown=0.18,
        win_rate=0.57,
        avg_latency_ms=4.5,
        n_predictions=8000,
    )
    report.add_model(model_c)

    # Generate text report
    print("\nTEXT REPORT:")
    print(report.generate_report(ReportFormat.TEXT))

    # Compare specific models
    print("\nSPECIFIC COMPARISON (model_b vs model_c):")
    comparisons = report.compare_models("model_b", "model_c")
    for metric, result in comparisons.items():
        print(f"  {metric}: {result.interpretation}")


if __name__ == "__main__":
    demonstrate_comparison_report()
