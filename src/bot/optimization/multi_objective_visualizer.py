"""
Multi-Objective Optimization Visualization Tools.
Provides comprehensive visualization of Pareto fronts, objective trade-offs, and evolution progress.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from bot.optimization.multi_objective import ParetoSolution

logger = logging.getLogger(__name__)


class MultiObjectiveVisualizer:
    """Visualization tools for multi-objective optimization results."""

    def __init__(self, output_dir: str = "outputs/multi_objective") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set style
        plt.style.use("seaborn-v0_8")
        sns.set_palette("husl")

    def plot_pareto_front(
        self,
        pareto_front: list[ParetoSolution],
        objectives: list[str] = None,
        save_path: str | None = None,
    ) -> None:
        """Plot the Pareto front showing trade-offs between objectives."""
        if not pareto_front:
            logger.warning("No Pareto front solutions to plot")
            return

        if objectives is None:
            objectives = ["sharpe_ratio", "max_drawdown", "consistency", "novelty", "robustness"]

        # Extract objective values
        objective_values = {obj: [] for obj in objectives}
        for solution in pareto_front:
            fitness_list = solution.fitness.to_list()
            for i, obj in enumerate(objectives):
                if i < len(fitness_list):
                    objective_values[obj].append(fitness_list[i])

        # Create subplots for pairwise comparisons
        n_objectives = len(objectives)
        fig, axes = plt.subplots(n_objectives, n_objectives, figsize=(15, 15))

        if n_objectives == 1:
            axes = [axes]
        elif n_objectives == 2:
            axes = axes.reshape(1, -1)

        for i, obj1 in enumerate(objectives):
            for j, obj2 in enumerate(objectives):
                if i == j:
                    # Diagonal: histogram of single objective
                    axes[i, j].hist(objective_values[obj1], bins=20, alpha=0.7, edgecolor="black")
                    axes[i, j].set_title(f"{obj1} Distribution")
                    axes[i, j].set_xlabel(obj1)
                    axes[i, j].set_ylabel("Frequency")
                else:
                    # Off-diagonal: scatter plot of two objectives
                    axes[i, j].scatter(
                        objective_values[obj2],
                        objective_values[obj1],
                        alpha=0.7,
                        s=50,
                        edgecolors="black",
                    )
                    axes[i, j].set_xlabel(obj2)
                    axes[i, j].set_ylabel(obj1)

                    # Add trend line
                    if len(objective_values[obj1]) > 1:
                        z = np.polyfit(objective_values[obj2], objective_values[obj1], 1)
                        p = np.poly1d(z)
                        axes[i, j].plot(
                            objective_values[obj2], p(objective_values[obj2]), "r--", alpha=0.8
                        )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Pareto front plot saved to {save_path}")

        plt.show()

    def plot_3d_pareto_front(
        self,
        pareto_front: list[ParetoSolution],
        objectives: list[str] = None,
        save_path: str | None = None,
    ) -> None:
        """Plot 3D Pareto front for three objectives."""
        if not pareto_front:
            logger.warning("No Pareto front solutions to plot")
            return

        if objectives is None:
            objectives = ["sharpe_ratio", "consistency", "novelty"]

        if len(objectives) != 3:
            logger.error("3D plot requires exactly 3 objectives")
            return

        # Extract objective values
        obj1_values = []
        obj2_values = []
        obj3_values = []

        for solution in pareto_front:
            fitness_list = solution.fitness.to_list()
            if len(fitness_list) >= 3:
                obj1_values.append(fitness_list[0])  # sharpe_ratio
                obj2_values.append(fitness_list[2])  # consistency
                obj3_values.append(fitness_list[3])  # novelty

        # Create 3D plot
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection="3d")

        scatter = ax.scatter(
            obj1_values, obj2_values, obj3_values, c=obj1_values, cmap="viridis", s=50, alpha=0.7
        )

        ax.set_xlabel(objectives[0])
        ax.set_ylabel(objectives[1])
        ax.set_zlabel(objectives[2])
        ax.set_title("3D Pareto Front")

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=20)
        cbar.set_label(objectives[0])

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"3D Pareto front plot saved to {save_path}")

        plt.show()

    def plot_evolution_progress(
        self, performance_history: dict[str, list[float]], save_path: str | None = None
    ) -> None:
        """Plot evolution progress over generations."""
        if not performance_history:
            logger.warning("No performance history to plot")
            return

        generations = performance_history.get("generation", [])
        if not generations:
            logger.warning("No generation data found")
            return

        # Create subplots for different metrics
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Pareto front size
        if "pareto_front_size" in performance_history:
            axes[0, 0].plot(
                generations,
                performance_history["pareto_front_size"],
                marker="o",
                linewidth=2,
                markersize=4,
            )
            axes[0, 0].set_title("Pareto Front Size")
            axes[0, 0].set_xlabel("Generation")
            axes[0, 0].set_ylabel("Number of Solutions")
            axes[0, 0].grid(True, alpha=0.3)

        # Average rank
        if "avg_rank" in performance_history:
            axes[0, 1].plot(
                generations,
                performance_history["avg_rank"],
                marker="s",
                linewidth=2,
                markersize=4,
                color="orange",
            )
            axes[0, 1].set_title("Average Rank")
            axes[0, 1].set_xlabel("Generation")
            axes[0, 1].set_ylabel("Average Rank")
            axes[0, 1].grid(True, alpha=0.3)

        # Best Sharpe ratio
        if "best_sharpe" in performance_history:
            axes[1, 0].plot(
                generations,
                performance_history["best_sharpe"],
                marker="^",
                linewidth=2,
                markersize=4,
                color="green",
            )
            axes[1, 0].set_title("Best Sharpe Ratio")
            axes[1, 0].set_xlabel("Generation")
            axes[1, 0].set_ylabel("Sharpe Ratio")
            axes[1, 0].grid(True, alpha=0.3)

        # Best consistency
        if "best_consistency" in performance_history:
            axes[1, 1].plot(
                generations,
                performance_history["best_consistency"],
                marker="d",
                linewidth=2,
                markersize=4,
                color="red",
            )
            axes[1, 1].set_title("Best Consistency")
            axes[1, 1].set_xlabel("Generation")
            axes[1, 1].set_ylabel("Consistency Score")
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Evolution progress plot saved to {save_path}")

        plt.show()

    def plot_objective_correlations(
        self, pareto_front: list[ParetoSolution], save_path: str | None = None
    ) -> None:
        """Plot correlation matrix between objectives."""
        if not pareto_front:
            logger.warning("No Pareto front solutions to plot")
            return

        # Extract all objective values
        objectives = ["sharpe_ratio", "max_drawdown", "consistency", "novelty", "robustness"]
        objective_data = {obj: [] for obj in objectives}

        for solution in pareto_front:
            fitness_list = solution.fitness.to_list()
            for i, obj in enumerate(objectives):
                if i < len(fitness_list):
                    objective_data[obj].append(fitness_list[i])

        # Create correlation matrix
        df = pd.DataFrame(objective_data)
        correlation_matrix = df.corr()

        # Plot correlation heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            correlation_matrix,
            annot=True,
            cmap="coolwarm",
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8},
        )
        plt.title("Objective Correlation Matrix")

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Correlation matrix plot saved to {save_path}")

        plt.show()

    def plot_solution_diversity(
        self, pareto_front: list[ParetoSolution], save_path: str | None = None
    ) -> None:
        """Plot diversity analysis of Pareto front solutions."""
        if not pareto_front:
            logger.warning("No Pareto front solutions to plot")
            return

        # Extract parameter values for diversity analysis
        param_names = list(pareto_front[0].parameters.keys())
        param_data = {param: [] for param in param_names}

        for solution in pareto_front:
            for param in param_names:
                param_data[param].append(solution.parameters[param])

        # Create subplots for parameter distributions
        n_params = len(param_names)
        n_cols = 4
        n_rows = (n_params + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)

        for i, param in enumerate(param_names):
            row = i // n_cols
            col = i % n_cols

            values = param_data[param]

            # Determine plot type based on data type
            if isinstance(values[0], bool):
                # Boolean: bar plot
                true_count = sum(values)
                false_count = len(values) - true_count
                axes[row, col].bar(
                    ["False", "True"], [false_count, true_count], color=["lightcoral", "lightblue"]
                )
                axes[row, col].set_title(f"{param} Distribution")
                axes[row, col].set_ylabel("Count")
            elif isinstance(values[0], str):
                # String: bar plot
                unique_values, counts = np.unique(values, return_counts=True)
                axes[row, col].bar(unique_values, counts, color="lightgreen")
                axes[row, col].set_title(f"{param} Distribution")
                axes[row, col].set_ylabel("Count")
                axes[row, col].tick_params(axis="x", rotation=45)
            else:
                # Numeric: histogram
                axes[row, col].hist(
                    values, bins=20, alpha=0.7, edgecolor="black", color="lightblue"
                )
                axes[row, col].set_title(f"{param} Distribution")
                axes[row, col].set_xlabel(param)
                axes[row, col].set_ylabel("Frequency")

        # Hide empty subplots
        for i in range(n_params, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            axes[row, col].set_visible(False)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Diversity analysis plot saved to {save_path}")

        plt.show()

    def plot_best_solutions_comparison(
        self, best_solutions: dict[str, ParetoSolution], save_path: str | None = None
    ) -> None:
        """Plot comparison of best solutions for each objective."""
        if not best_solutions:
            logger.warning("No best solutions to plot")
            return

        objectives = list(best_solutions.keys())
        n_objectives = len(objectives)

        # Extract fitness values
        fitness_data = []
        solution_names = []

        for obj, solution in best_solutions.items():
            fitness_data.append(solution.fitness.to_list())
            solution_names.append(f"Best {obj}")

        # Create radar chart
        angles = np.linspace(0, 2 * np.pi, len(fitness_data[0]), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection="polar"))

        colors = plt.cm.Set3(np.linspace(0, 1, n_objectives))

        for _i, (fitness_values, name, color) in enumerate(
            zip(fitness_data, solution_names, colors, strict=False)
        ):
            # Normalize values to 0-1 for radar chart
            normalized_values = [
                (
                    (v - min(fitness_values)) / (max(fitness_values) - min(fitness_values))
                    if max(fitness_values) != min(fitness_values)
                    else 0.5
                )
                for v in fitness_values
            ]
            normalized_values += normalized_values[:1]  # Complete the circle

            ax.plot(angles, normalized_values, "o-", linewidth=2, label=name, color=color)
            ax.fill(angles, normalized_values, alpha=0.25, color=color)

        # Set labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(["Sharpe", "Drawdown", "Consistency", "Novelty", "Robustness"])
        ax.set_ylim(0, 1)
        ax.set_title("Best Solutions Comparison", size=16, y=1.08)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Best solutions comparison plot saved to {save_path}")

        plt.show()

    def create_comprehensive_report(
        self, results: dict[str, Any], save_dir: str | None = None
    ) -> None:
        """Create a comprehensive visualization report."""
        if save_dir is None:
            save_dir = self.output_dir

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        pareto_front = results.get("pareto_front", [])
        performance_history = results.get("performance_history", {})
        best_solutions = results.get("best_solutions", {})

        # Generate all plots
        self.plot_pareto_front(pareto_front, save_path=save_dir / "pareto_front.png")

        self.plot_3d_pareto_front(pareto_front, save_path=save_dir / "pareto_front_3d.png")

        self.plot_evolution_progress(
            performance_history, save_path=save_dir / "evolution_progress.png"
        )

        self.plot_objective_correlations(
            pareto_front, save_path=save_dir / "objective_correlations.png"
        )

        self.plot_solution_diversity(pareto_front, save_path=save_dir / "solution_diversity.png")

        if best_solutions:
            self.plot_best_solutions_comparison(
                best_solutions, save_path=save_dir / "best_solutions_comparison.png"
            )

        # Create summary report
        self._create_summary_report(results, save_dir / "summary_report.txt")

        logger.info(f"Comprehensive report saved to {save_dir}")

    def _create_summary_report(self, results: dict[str, Any], save_path: Path) -> None:
        """Create a text summary report."""
        pareto_front = results.get("pareto_front", [])
        performance_history = results.get("performance_history", {})
        best_solutions = results.get("best_solutions", {})
        diversity_analysis = results.get("diversity_analysis", {})

        with open(save_path, "w") as f:
            f.write("Multi-Objective Optimization Summary Report\n")
            f.write("=" * 50 + "\n\n")

            f.write(f"Pareto Front Size: {len(pareto_front)}\n")
            f.write(f"Generations Completed: {len(performance_history.get('generation', []))}\n\n")

            if pareto_front:
                f.write("Pareto Front Statistics:\n")
                f.write("-" * 25 + "\n")

                objectives = [
                    "sharpe_ratio",
                    "max_drawdown",
                    "consistency",
                    "novelty",
                    "robustness",
                ]
                for i, obj in enumerate(objectives):
                    values = [s.fitness.to_list()[i] for s in pareto_front]
                    f.write(f"{obj}:\n")
                    f.write(f"  Min: {min(values):.4f}\n")
                    f.write(f"  Max: {max(values):.4f}\n")
                    f.write(f"  Mean: {np.mean(values):.4f}\n")
                    f.write(f"  Std: {np.std(values):.4f}\n\n")

            if best_solutions:
                f.write("Best Solutions:\n")
                f.write("-" * 15 + "\n")
                for obj, solution in best_solutions.items():
                    f.write(f"Best {obj}:\n")
                    f.write(f"  Sharpe: {solution.fitness.sharpe_ratio:.4f}\n")
                    f.write(f"  Max Drawdown: {solution.fitness.max_drawdown:.4f}\n")
                    f.write(f"  Consistency: {solution.fitness.consistency:.4f}\n")
                    f.write(f"  Novelty: {solution.fitness.novelty:.4f}\n")
                    f.write(f"  Robustness: {solution.fitness.robustness:.4f}\n\n")

            if diversity_analysis:
                f.write("Diversity Analysis:\n")
                f.write("-" * 20 + "\n")
                f.write(f"Diversity Score: {diversity_analysis.get('diversity_score', 0):.4f}\n")

                solution_types = diversity_analysis.get("solution_types", {})
                if solution_types:
                    f.write("Solution Types:\n")
                    for sol_type, count in solution_types.items():
                        f.write(f"  {sol_type}: {count}\n")

        logger.info(f"Summary report saved to {save_path}")
