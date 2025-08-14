"""
Multi-Objective Optimization System for Strategy Discovery.
Implements Pareto front identification, non-dominated sorting, and multi-objective selection.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from bot.optimization.config import OptimizationConfig, StrategyConfig
from bot.optimization.strategy_diversity import StrategyDiversityTracker

logger = logging.getLogger(__name__)


@dataclass
class MultiObjectiveFitness:
    """Multi-objective fitness scores for a strategy."""

    sharpe_ratio: float
    max_drawdown: float  # Negative value (lower is better)
    consistency: float  # 0-1 score
    novelty: float  # 0-1 score
    robustness: float  # 0-1 score

    def to_list(self) -> list[float]:
        """Convert to list for numpy operations."""
        return [
            self.sharpe_ratio,
            -self.max_drawdown,
            self.consistency,
            self.novelty,
            self.robustness,
        ]

    def dominates(self, other: MultiObjectiveFitness) -> bool:
        """Check if this fitness dominates another (Pareto dominance)."""
        self_values = self.to_list()
        other_values = other.to_list()

        # At least one objective is better
        at_least_one_better = any(s > o for s, o in zip(self_values, other_values, strict=False))
        # No objective is worse
        no_objective_worse = all(s >= o for s, o in zip(self_values, other_values, strict=False))

        return at_least_one_better and no_objective_worse


@dataclass
class ParetoSolution:
    """A solution on the Pareto front."""

    parameters: dict[str, Any]
    fitness: MultiObjectiveFitness
    rank: int
    crowding_distance: float = 0.0
    generation: int = 0


class MultiObjectiveEvolution:
    """Multi-objective evolutionary optimization using NSGA-II algorithm."""

    def __init__(self, config: OptimizationConfig, strategy_config: StrategyConfig) -> None:
        self.config = config
        self.strategy_config = strategy_config
        self.population: list[ParetoSolution] = []
        self.pareto_front: list[ParetoSolution] = []
        self.generation_history: list[dict[str, Any]] = []

        # Multi-objective configuration
        self.objectives = ["sharpe_ratio", "max_drawdown", "consistency", "novelty", "robustness"]

        # NSGA-II parameters
        self.tournament_size = 3
        self.crossover_rate = 0.8
        self.mutation_rate = 0.2

        # Diversity tracking
        self.diversity_tracker = StrategyDiversityTracker(
            output_dir=Path(config.output_dir), strategy_config=strategy_config
        )

        # Performance tracking
        self.performance_history = {
            "generation": [],
            "pareto_front_size": [],
            "avg_rank": [],
            "diversity_score": [],
            "best_sharpe": [],
            "best_consistency": [],
            "best_novelty": [],
        }

    def evolve(
        self,
        evaluate_func: Callable[[dict[str, Any]], dict[str, Any]],
        generations: int = 100,
        population_size: int = 50,
    ) -> dict[str, Any]:
        """Run multi-objective evolutionary optimization."""
        logger.info(
            f"Starting multi-objective evolution: {generations} generations, {population_size} population"
        )

        # Initialize population
        self._initialize_population(population_size)

        # Evolution loop
        for generation in range(generations):
            logger.info(f"Generation {generation + 1}/{generations}")

            # Evaluate population
            self._evaluate_population(evaluate_func)

            # Non-dominated sorting
            self._non_dominated_sorting()

            # Calculate crowding distance
            self._calculate_crowding_distance()

            # Update Pareto front
            self._update_pareto_front()

            # Track performance
            self._track_performance(generation)

            # Selection and reproduction
            self._evolve_population()

            # Log progress
            if generation % 10 == 0:
                self._log_progress(generation)

        return self._generate_results()

    def _initialize_population(self, size: int) -> None:
        """Initialize random population."""
        self.population = []
        for _ in range(size):
            parameters = self._generate_random_parameters()
            solution = ParetoSolution(
                parameters=parameters,
                fitness=MultiObjectiveFitness(0.0, 0.0, 0.0, 0.0, 0.0),
                rank=0,
            )
            self.population.append(solution)

        logger.info(f"Initialized population of {size} individuals")

    def _generate_random_parameters(self) -> dict[str, Any]:
        """Generate random strategy parameters."""
        return {
            "donchian_lookback": np.random.randint(10, 200),
            "atr_period": np.random.randint(5, 50),
            "atr_k": np.random.uniform(0.5, 5.0),
            "volume_ma_period": np.random.randint(10, 50),
            "volume_threshold": np.random.uniform(1.0, 3.0),
            "use_volume_filter": np.random.choice([True, False]),
            "rsi_period": np.random.randint(10, 30),
            "rsi_oversold": np.random.uniform(20.0, 40.0),
            "rsi_overbought": np.random.uniform(60.0, 80.0),
            "use_rsi_filter": np.random.choice([True, False]),
            "bollinger_period": np.random.randint(10, 50),
            "bollinger_std": np.random.uniform(1.5, 3.0),
            "use_bollinger_filter": np.random.choice([True, False]),
            "entry_confirmation_periods": np.random.randint(0, 5),
            "exit_confirmation_periods": np.random.randint(0, 5),
            "cooldown_periods": np.random.randint(0, 10),
            "max_risk_per_trade": np.random.uniform(0.01, 0.05),
            "position_sizing_method": np.random.choice(["atr", "fixed", "kelly"]),
            "use_regime_filter": np.random.choice([True, False]),
            "regime_lookback": np.random.randint(100, 300),
            "use_correlation_filter": np.random.choice([True, False]),
            "correlation_threshold": np.random.uniform(0.5, 0.9),
            "correlation_lookback": np.random.randint(30, 100),
        }

    def _evaluate_population(
        self, evaluate_func: Callable[[dict[str, Any]], dict[str, Any]]
    ) -> None:
        """Evaluate all individuals in the population."""
        for solution in self.population:
            try:
                result = evaluate_func(solution.parameters)

                # Calculate multi-objective fitness
                fitness = MultiObjectiveFitness(
                    sharpe_ratio=result.get("sharpe", 0.0),
                    max_drawdown=result.get("max_drawdown", 0.0),
                    consistency=self._calculate_consistency(result),
                    novelty=self._calculate_novelty(solution.parameters),
                    robustness=self._calculate_robustness(result),
                )

                solution.fitness = fitness

            except Exception as e:
                logger.warning(f"Evaluation failed: {e}")
                solution.fitness = MultiObjectiveFitness(0.0, 0.0, 0.0, 0.0, 0.0)

    def _calculate_consistency(self, result: dict[str, Any]) -> float:
        """Calculate consistency score based on performance metrics."""
        # Simple consistency based on win rate and trade frequency
        win_rate = result.get("win_rate", 0.5)
        n_trades = result.get("n_trades", 0)

        # Consistency increases with win rate and reasonable trade frequency
        trade_consistency = min(1.0, n_trades / 100.0)  # Normalize to 0-1
        return (win_rate + trade_consistency) / 2.0

    def _calculate_novelty(self, parameters: dict[str, Any]) -> float:
        """Calculate novelty score based on parameter uniqueness."""
        if not self.population:
            return 1.0

        # Calculate average distance to other solutions
        distances = []
        for other in self.population:
            if other.parameters != parameters:
                distance = self._calculate_parameter_distance(parameters, other.parameters)
                distances.append(distance)

        return np.mean(distances) if distances else 1.0

    def _calculate_robustness(self, result: dict[str, Any]) -> float:
        """Calculate robustness score based on performance stability."""
        # Simple robustness based on drawdown and Sharpe ratio
        sharpe = result.get("sharpe", 0.0)
        max_dd = result.get("max_drawdown", 1.0)

        # Robustness increases with higher Sharpe and lower drawdown
        sharpe_score = max(0.0, min(1.0, sharpe / 2.0))  # Normalize to 0-1
        dd_score = max(0.0, 1.0 - max_dd)  # Lower drawdown is better

        return (sharpe_score + dd_score) / 2.0

    def _calculate_parameter_distance(
        self, params1: dict[str, Any], params2: dict[str, Any]
    ) -> float:
        """Calculate normalized distance between parameter sets."""
        total_distance = 0.0
        count = 0

        for key in params1:
            if key in params2:
                val1 = params1[key]
                val2 = params2[key]

                if isinstance(val1, int | float) and isinstance(val2, int | float):
                    # Normalize numeric parameters
                    if key in ["donchian_lookback", "atr_period", "volume_ma_period"]:
                        max_val = 200.0
                    elif key in ["atr_k", "volume_threshold", "bollinger_std"]:
                        max_val = 5.0
                    else:
                        max_val = 1.0

                    distance = abs(val1 - val2) / max_val
                    total_distance += distance
                    count += 1
                elif isinstance(val1, bool) and isinstance(val2, bool):
                    # Boolean parameters
                    distance = 0.0 if val1 == val2 else 1.0
                    total_distance += distance
                    count += 1

        return total_distance / count if count > 0 else 0.0

    def _non_dominated_sorting(self) -> None:
        """Perform non-dominated sorting to assign ranks."""
        # Calculate domination counts and dominated sets
        domination_counts = [0] * len(self.population)
        dominated_sets = [[] for _ in range(len(self.population))]

        for i, solution1 in enumerate(self.population):
            for j, solution2 in enumerate(self.population):
                if i != j:
                    if solution1.fitness.dominates(solution2.fitness):
                        dominated_sets[i].append(j)
                    elif solution2.fitness.dominates(solution1.fitness):
                        domination_counts[i] += 1

        # Assign ranks
        current_rank = 0
        remaining = set(range(len(self.population)))

        while remaining:
            current_front = []
            for i in remaining:
                if domination_counts[i] == 0:
                    current_front.append(i)
                    self.population[i].rank = current_rank

            # Remove current front from remaining
            for i in current_front:
                remaining.remove(i)
                # Update domination counts for dominated solutions
                for j in dominated_sets[i]:
                    domination_counts[j] -= 1

            current_rank += 1

    def _calculate_crowding_distance(self) -> None:
        """Calculate crowding distance for diversity preservation."""
        # Group solutions by rank
        rank_groups = {}
        for i, solution in enumerate(self.population):
            rank = solution.rank
            if rank not in rank_groups:
                rank_groups[rank] = []
            rank_groups[rank].append(i)

        # Calculate crowding distance for each rank
        for rank, indices in rank_groups.items():
            if len(indices) <= 2:
                # Assign infinite distance to boundary solutions
                for idx in indices:
                    self.population[idx].crowding_distance = float("inf")
            else:
                # Calculate crowding distance for each objective
                for obj_idx in range(len(self.objectives)):
                    # Sort by objective value
                    sorted_indices = sorted(
                        indices, key=lambda i: self.population[i].fitness.to_list()[obj_idx]
                    )

                    # Boundary solutions get infinite distance
                    self.population[sorted_indices[0]].crowding_distance = float("inf")
                    self.population[sorted_indices[-1]].crowding_distance = float("inf")

                    # Calculate distance for interior solutions
                    obj_values = [
                        self.population[i].fitness.to_list()[obj_idx] for i in sorted_indices
                    ]
                    obj_range = obj_values[-1] - obj_values[0]

                    if obj_range > 0:
                        for i in range(1, len(sorted_indices) - 1):
                            distance = (obj_values[i + 1] - obj_values[i - 1]) / obj_range
                            self.population[sorted_indices[i]].crowding_distance += distance

    def _update_pareto_front(self) -> None:
        """Update the Pareto front with current non-dominated solutions."""
        rank_0_solutions = [s for s in self.population if s.rank == 0]

        # Add new solutions to Pareto front
        for solution in rank_0_solutions:
            # Check if solution is not dominated by any existing Pareto solution
            is_non_dominated = True
            for existing in self.pareto_front:
                if existing.fitness.dominates(solution.fitness):
                    is_non_dominated = False
                    break

            if is_non_dominated:
                # Remove dominated solutions from Pareto front
                self.pareto_front = [
                    s for s in self.pareto_front if not solution.fitness.dominates(s.fitness)
                ]
                self.pareto_front.append(solution)

    def _evolve_population(self) -> None:
        """Create next generation using tournament selection and genetic operators."""
        new_population = []

        # Elitism: keep best solutions (rank 0)
        elite_solutions = [s for s in self.population if s.rank == 0]
        elite_size = min(len(elite_solutions), len(self.population) // 2)

        # Sort by crowding distance for diversity
        elite_solutions.sort(key=lambda s: s.crowding_distance, reverse=True)
        new_population.extend(elite_solutions[:elite_size])

        # Generate offspring
        while len(new_population) < len(self.population):
            # Tournament selection
            parent1 = self._tournament_selection()
            parent2 = self._tournament_selection()

            # Crossover
            if np.random.random() < self.crossover_rate:
                child_params = self._crossover(parent1.parameters, parent2.parameters)
            else:
                child_params = parent1.parameters.copy()

            # Mutation
            if np.random.random() < self.mutation_rate:
                child_params = self._mutate(child_params)

            # Create child solution
            child = ParetoSolution(
                parameters=child_params,
                fitness=MultiObjectiveFitness(0.0, 0.0, 0.0, 0.0, 0.0),
                rank=0,
            )
            new_population.append(child)

        self.population = new_population

    def _tournament_selection(self) -> ParetoSolution:
        """Tournament selection based on rank and crowding distance."""
        tournament = np.random.choice(self.population, self.tournament_size, replace=False)

        # Select based on rank first, then crowding distance
        best = tournament[0]
        for solution in tournament[1:]:
            if solution.rank < best.rank:
                best = solution
            elif solution.rank == best.rank and solution.crowding_distance > best.crowding_distance:
                best = solution

        return best

    def _crossover(self, params1: dict[str, Any], params2: dict[str, Any]) -> dict[str, Any]:
        """Uniform crossover between two parameter sets."""
        child_params = {}

        for key in params1:
            if np.random.random() < 0.5:
                child_params[key] = params1[key]
            else:
                child_params[key] = params2[key]

        return child_params

    def _mutate(self, parameters: dict[str, Any]) -> dict[str, Any]:
        """Mutate parameters with small random changes."""
        mutated = parameters.copy()

        for key, value in mutated.items():
            if np.random.random() < 0.1:  # 10% mutation probability per parameter
                if isinstance(value, int):
                    if key in ["donchian_lookback", "atr_period", "volume_ma_period"]:
                        mutated[key] = max(1, value + np.random.randint(-10, 11))
                    elif key in ["rsi_period", "bollinger_period"]:
                        mutated[key] = max(1, value + np.random.randint(-5, 6))
                    else:
                        mutated[key] = max(0, value + np.random.randint(-2, 3))
                elif isinstance(value, float):
                    if key in ["atr_k", "volume_threshold", "bollinger_std"]:
                        mutated[key] = max(0.1, value + np.random.uniform(-0.5, 0.5))
                    elif key in ["rsi_oversold", "rsi_overbought"]:
                        mutated[key] = np.clip(value + np.random.uniform(-5, 5), 0, 100)
                    else:
                        mutated[key] = max(0.0, value + np.random.uniform(-0.1, 0.1))
                elif isinstance(value, bool):
                    mutated[key] = not value

        return mutated

    def _track_performance(self, generation: int) -> None:
        """Track performance metrics for the generation."""
        self.performance_history["generation"].append(generation)
        self.performance_history["pareto_front_size"].append(len(self.pareto_front))

        if self.population:
            avg_rank = np.mean([s.rank for s in self.population])
            self.performance_history["avg_rank"].append(avg_rank)

            best_sharpe = max([s.fitness.sharpe_ratio for s in self.population])
            best_consistency = max([s.fitness.consistency for s in self.population])
            best_novelty = max([s.fitness.novelty for s in self.population])

            self.performance_history["best_sharpe"].append(best_sharpe)
            self.performance_history["best_consistency"].append(best_consistency)
            self.performance_history["best_novelty"].append(best_novelty)

    def _log_progress(self, generation: int) -> None:
        """Log progress information."""
        if self.pareto_front:
            best_sharpe = max([s.fitness.sharpe_ratio for s in self.pareto_front])
            best_consistency = max([s.fitness.consistency for s in self.pareto_front])
            best_novelty = max([s.fitness.novelty for s in self.pareto_front])

            logger.info(
                f"Generation {generation}: Pareto front size={len(self.pareto_front)}, "
                f"Best Sharpe={best_sharpe:.3f}, Best Consistency={best_consistency:.3f}, "
                f"Best Novelty={best_novelty:.3f}"
            )

    def _generate_results(self) -> dict[str, Any]:
        """Generate final results."""
        return {
            "pareto_front": self.pareto_front,
            "pareto_front_size": len(self.pareto_front),
            "generations_completed": len(self.performance_history["generation"]),
            "performance_history": self.performance_history,
            "best_solutions": self._get_best_solutions(),
            "diversity_analysis": self._analyze_diversity(),
        }

    def _get_best_solutions(self) -> dict[str, ParetoSolution]:
        """Get best solutions for each objective."""
        if not self.pareto_front:
            return {}

        best_solutions = {}
        objectives = ["sharpe_ratio", "max_drawdown", "consistency", "novelty", "robustness"]

        for i, objective in enumerate(objectives):
            best_solution = max(self.pareto_front, key=lambda s: s.fitness.to_list()[i])
            best_solutions[objective] = best_solution

        return best_solutions

    def _analyze_diversity(self) -> dict[str, Any]:
        """Analyze diversity of the Pareto front."""
        if len(self.pareto_front) < 2:
            return {"diversity_score": 0.0, "solution_types": {}}

        # Calculate average distance between Pareto solutions
        distances = []
        for i, sol1 in enumerate(self.pareto_front):
            for _j, sol2 in enumerate(self.pareto_front[i + 1 :], i + 1):
                distance = self._calculate_parameter_distance(sol1.parameters, sol2.parameters)
                distances.append(distance)

        diversity_score = np.mean(distances) if distances else 0.0

        # Analyze solution types
        solution_types = self._identify_solution_types()

        return {
            "diversity_score": diversity_score,
            "solution_types": solution_types,
            "pareto_distribution": self._analyze_pareto_distribution(),
        }

    def _identify_solution_types(self) -> dict[str, int]:
        """Identify different types of solutions in the Pareto front."""
        types = {
            "high_sharpe": 0,
            "low_drawdown": 0,
            "high_consistency": 0,
            "high_novelty": 0,
            "balanced": 0,
        }

        for solution in self.pareto_front:
            fitness = solution.fitness

            if fitness.sharpe_ratio > 1.5:
                types["high_sharpe"] += 1
            elif fitness.max_drawdown < 0.1:
                types["low_drawdown"] += 1
            elif fitness.consistency > 0.8:
                types["high_consistency"] += 1
            elif fitness.novelty > 0.7:
                types["high_novelty"] += 1
            else:
                types["balanced"] += 1

        return types

    def _analyze_pareto_distribution(self) -> dict[str, float]:
        """Analyze the distribution of solutions across the Pareto front."""
        if not self.pareto_front:
            return {}

        # Calculate objective ranges
        objectives = ["sharpe_ratio", "max_drawdown", "consistency", "novelty", "robustness"]
        distributions = {}

        for i, objective in enumerate(objectives):
            values = [s.fitness.to_list()[i] for s in self.pareto_front]
            distributions[objective] = {
                "min": min(values),
                "max": max(values),
                "mean": np.mean(values),
                "std": np.std(values),
            }

        return distributions
