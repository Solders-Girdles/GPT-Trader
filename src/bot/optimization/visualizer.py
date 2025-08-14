"""
Visualization tools for optimization results.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)


class ResultVisualizer:
    """Creates visualizations for optimization results."""

    def __init__(self) -> None:
        # Set style
        plt.style.use("seaborn-v0_8")
        sns.set_palette("husl")

    def plot_parameter_sensitivity(self, results: list[dict[str, Any]], output_dir: Path) -> None:
        """Create parameter sensitivity plots."""
        if not results:
            logger.warning("No results to visualize")
            return

        # Convert to DataFrame
        df = self._results_to_dataframe(results)

        # Get parameter columns
        param_cols = [col for col in df.columns if col.startswith("param_")]

        if not param_cols or "sharpe" not in df.columns:
            logger.warning("No parameters or Sharpe ratio found for sensitivity plot")
            return

        # Create subplots
        n_params = len(param_cols)
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()

        for i, param_col in enumerate(param_cols[:4]):  # Limit to 4 parameters
            param_name = param_col.replace("param_", "")

            # Scatter plot
            axes[i].scatter(df[param_col], df["sharpe"], alpha=0.6, s=20)
            axes[i].set_xlabel(param_name)
            axes[i].set_ylabel("Sharpe Ratio")
            axes[i].set_title(f"Parameter Sensitivity: {param_name}")

            # Add trend line
            z = np.polyfit(df[param_col].dropna(), df["sharpe"].dropna(), 1)
            p = np.poly1d(z)
            axes[i].plot(df[param_col], p(df[param_col]), "r--", alpha=0.8)

        # Hide unused subplots
        for i in range(n_params, 4):
            axes[i].set_visible(False)

        plt.tight_layout()
        plt.savefig(output_dir / "parameter_sensitivity.png", dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(
            f"Parameter sensitivity plot saved to {output_dir / 'parameter_sensitivity.png'}"
        )

    def plot_optimization_progress(self, results: list[dict[str, Any]], output_dir: Path) -> None:
        """Create optimization progress plots."""
        if not results:
            logger.warning("No results to visualize")
            return

        # Convert to DataFrame
        df = self._results_to_dataframe(results)

        if "sharpe" not in df.columns:
            logger.warning("Sharpe ratio not found for progress plot")
            return

        # Create progress plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # Plot 1: Sharpe ratio distribution
        ax1.hist(df["sharpe"].dropna(), bins=30, alpha=0.7, edgecolor="black")
        ax1.axvline(
            df["sharpe"].mean(),
            color="red",
            linestyle="--",
            label=f'Mean: {df["sharpe"].mean():.3f}',
        )
        ax1.axvline(
            df["sharpe"].max(),
            color="green",
            linestyle="--",
            label=f'Max: {df["sharpe"].max():.3f}',
        )
        ax1.set_xlabel("Sharpe Ratio")
        ax1.set_ylabel("Frequency")
        ax1.set_title("Distribution of Sharpe Ratios")
        ax1.legend()

        # Plot 2: Cumulative best performance
        sorted_sharpe = df["sharpe"].sort_values(ascending=False)
        cumulative_best = sorted_sharpe.cummax()

        ax2.plot(range(len(cumulative_best)), cumulative_best, linewidth=2)
        ax2.set_xlabel("Evaluation Number")
        ax2.set_ylabel("Best Sharpe Ratio")
        ax2.set_title("Optimization Progress")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / "optimization_progress.png", dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(
            f"Optimization progress plot saved to {output_dir / 'optimization_progress.png'}"
        )

    def plot_correlation_matrix(self, results: list[dict[str, Any]], output_dir: Path) -> None:
        """Create correlation matrix heatmap."""
        if not results:
            logger.warning("No results to visualize")
            return

        # Convert to DataFrame
        df = self._results_to_dataframe(results)

        # Get parameter and metric columns
        param_cols = [col for col in df.columns if col.startswith("param_")]
        metric_cols = ["sharpe", "cagr", "max_drawdown", "total_return"]

        # Filter to existing columns
        param_cols = [col for col in param_cols if col in df.columns]
        metric_cols = [col for col in metric_cols if col in df.columns]

        if not param_cols or not metric_cols:
            logger.warning("Insufficient data for correlation matrix")
            return

        # Create correlation matrix
        corr_data = df[param_cols + metric_cols].corr()

        # Create heatmap
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_data, dtype=bool))

        sns.heatmap(
            corr_data,
            mask=mask,
            annot=True,
            cmap="coolwarm",
            center=0,
            square=True,
            fmt=".2f",
            cbar_kws={"shrink": 0.8},
        )

        plt.title("Parameter and Metric Correlations")
        plt.tight_layout()
        plt.savefig(output_dir / "correlation_matrix.png", dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Correlation matrix saved to {output_dir / 'correlation_matrix.png'}")

    def plot_parameter_distributions(self, results: list[dict[str, Any]], output_dir: Path) -> None:
        """Create parameter distribution plots."""
        if not results:
            logger.warning("No results to visualize")
            return

        # Convert to DataFrame
        df = self._results_to_dataframe(results)

        # Get parameter columns
        param_cols = [col for col in df.columns if col.startswith("param_")]

        if not param_cols:
            logger.warning("No parameters found for distribution plot")
            return

        # Create subplots
        n_params = len(param_cols)
        n_cols = min(3, n_params)
        n_rows = (n_params + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        if n_params == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)

        for i, param_col in enumerate(param_cols):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col]

            param_name = param_col.replace("param_", "")
            values = df[param_col].dropna()

            # Create histogram
            ax.hist(values, bins=20, alpha=0.7, edgecolor="black")
            ax.set_xlabel(param_name)
            ax.set_ylabel("Frequency")
            ax.set_title(f"Distribution: {param_name}")

            # Add statistics
            mean_val = values.mean()
            ax.axvline(mean_val, color="red", linestyle="--", label=f"Mean: {mean_val:.2f}")
            ax.legend()

        # Hide unused subplots
        for i in range(n_params, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            axes[row, col].set_visible(False)

        plt.tight_layout()
        plt.savefig(output_dir / "parameter_distributions.png", dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(
            f"Parameter distributions saved to {output_dir / 'parameter_distributions.png'}"
        )

    def plot_performance_scatter(self, results: list[dict[str, Any]], output_dir: Path) -> None:
        """Create performance scatter plots."""
        if not results:
            logger.warning("No results to visualize")
            return

        # Convert to DataFrame
        df = self._results_to_dataframe(results)

        # Check for required metrics
        required_metrics = ["sharpe", "cagr", "max_drawdown"]
        available_metrics = [m for m in required_metrics if m in df.columns]

        if len(available_metrics) < 2:
            logger.warning("Insufficient metrics for scatter plot")
            return

        # Create scatter plots
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Plot 1: Sharpe vs CAGR
        if "sharpe" in available_metrics and "cagr" in available_metrics:
            scatter = axes[0].scatter(
                df["sharpe"],
                df["cagr"],
                c=df["max_drawdown"] if "max_drawdown" in available_metrics else None,
                alpha=0.6,
                s=30,
            )
            axes[0].set_xlabel("Sharpe Ratio")
            axes[0].set_ylabel("CAGR")
            axes[0].set_title("Sharpe vs CAGR")
            if "max_drawdown" in available_metrics:
                plt.colorbar(scatter, ax=axes[0], label="Max Drawdown")

        # Plot 2: Sharpe vs Max Drawdown
        if "sharpe" in available_metrics and "max_drawdown" in available_metrics:
            scatter = axes[1].scatter(
                df["sharpe"],
                df["max_drawdown"],
                c=df["cagr"] if "cagr" in available_metrics else None,
                alpha=0.6,
                s=30,
            )
            axes[1].set_xlabel("Sharpe Ratio")
            axes[1].set_ylabel("Max Drawdown")
            axes[1].set_title("Sharpe vs Max Drawdown")
            if "cagr" in available_metrics:
                plt.colorbar(scatter, ax=axes[1], label="CAGR")

        plt.tight_layout()
        plt.savefig(output_dir / "performance_scatter.png", dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Performance scatter plot saved to {output_dir / 'performance_scatter.png'}")

    def create_dashboard(self, results: list[dict[str, Any]], output_dir: Path) -> None:
        """Create a comprehensive dashboard with all visualizations."""
        logger.info("Creating optimization dashboard...")

        # Create all plots
        self.plot_parameter_sensitivity(results, output_dir)
        self.plot_optimization_progress(results, output_dir)
        self.plot_correlation_matrix(results, output_dir)
        self.plot_parameter_distributions(results, output_dir)
        self.plot_performance_scatter(results, output_dir)

        # Create HTML dashboard
        self._create_html_dashboard(results, output_dir)

        logger.info(f"Dashboard created in {output_dir}")

    def _create_html_dashboard(self, results: list[dict[str, Any]], output_dir: Path) -> None:
        """Create an HTML dashboard with all visualizations."""
        html_content = (
            """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Optimization Results Dashboard</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .header { text-align: center; margin-bottom: 30px; }
                .plot-section { margin-bottom: 40px; }
                .plot-section h2 { color: #333; border-bottom: 2px solid #eee; padding-bottom: 10px; }
                .plot-container { text-align: center; margin: 20px 0; }
                .plot-container img { max-width: 100%; height: auto; border: 1px solid #ddd; }
                .stats { background: #f9f9f9; padding: 20px; border-radius: 5px; margin: 20px 0; }
                .stats h3 { margin-top: 0; }
                .stats table { width: 100%; border-collapse: collapse; }
                .stats th, .stats td { padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }
                .stats th { background-color: #f2f2f2; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Optimization Results Dashboard</h1>
                <p>Generated on """
            + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
            + """</p>
            </div>
        """
        )

        # Add statistics section
        if results:
            df = self._results_to_dataframe(results)
            if "sharpe" in df.columns:
                html_content += """
                <div class="stats">
                    <h3>Summary Statistics</h3>
                    <table>
                        <tr><th>Metric</th><th>Count</th><th>Mean</th><th>Std</th><th>Min</th><th>Max</th></tr>
                """

                for metric in ["sharpe", "cagr", "max_drawdown", "total_return"]:
                    if metric in df.columns:
                        values = df[metric].dropna()
                        if len(values) > 0:
                            html_content += f"""
                            <tr>
                                <td>{metric.title()}</td>
                                <td>{len(values)}</td>
                                <td>{values.mean():.4f}</td>
                                <td>{values.std():.4f}</td>
                                <td>{values.min():.4f}</td>
                                <td>{values.max():.4f}</td>
                            </tr>
                            """

                html_content += "</table></div>"

        # Add plot sections
        plot_files = [
            ("parameter_sensitivity.png", "Parameter Sensitivity Analysis"),
            ("optimization_progress.png", "Optimization Progress"),
            ("correlation_matrix.png", "Parameter and Metric Correlations"),
            ("parameter_distributions.png", "Parameter Distributions"),
            ("performance_scatter.png", "Performance Scatter Plots"),
        ]

        for filename, title in plot_files:
            plot_path = output_dir / filename
            if plot_path.exists():
                html_content += f"""
                <div class="plot-section">
                    <h2>{title}</h2>
                    <div class="plot-container">
                        <img src="{filename}" alt="{title}">
                    </div>
                </div>
                """

        html_content += """
        </body>
        </html>
        """

        # Save HTML file
        dashboard_path = output_dir / "dashboard.html"
        with open(dashboard_path, "w") as f:
            f.write(html_content)

        logger.info(f"HTML dashboard saved to {dashboard_path}")

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
