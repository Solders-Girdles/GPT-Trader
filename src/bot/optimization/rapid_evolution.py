"""
Rapid evolutionary optimization with minimal file I/O for fast iteration.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import datetime
from typing import Any

import numpy as np
from bot.optimization.config import OptimizationConfig, StrategyConfig

logger = logging.getLogger(__name__)


class RapidEvolutionEngine:
    """Fast evolutionary optimization with minimal file I/O."""

    def __init__(self, config: OptimizationConfig, strategy_config: StrategyConfig) -> None:
        self.config = config
        self.strategy_config = strategy_config
        self.population: list[dict[str, Any]] = []
        self.generation_history: list[dict[str, Any]] = []
        self.best_strategies: list[dict[str, Any]] = []
        self.diverse_strategies: list[dict[str, Any]] = []

        # Performance tracking
        self.performance_history = {
            "best_sharpe": [],
            "avg_sharpe": [],
            "diversity_score": [],
            "generation": [],
        }

    def initialize_population(self, size: int) -> None:
        """Initialize random population."""
        self.population = []
        for _ in range(size):
            individual = self._generate_random_individual()
            self.population.append(individual)
        logger.info(f"Initialized population of {size} individuals")

    def _generate_random_individual(self) -> dict[str, Any]:
        """Generate a random individual with parameters within bounds."""
        individual = {}
        for param_name, param_def in self.strategy_config.parameters.items():
            if param_def.type == "int":
                min_val = param_def.min_value or 0
                max_val = param_def.max_value or 100
                individual[param_name] = np.random.randint(min_val, max_val + 1)
            elif param_def.type == "float":
                min_val = param_def.min_value or 0.0
                max_val = param_def.max_value or 1.0
                individual[param_name] = np.random.uniform(min_val, max_val)
            elif param_def.type == "bool":
                individual[param_name] = np.random.choice([True, False])
            else:
                # Use default for other types
                individual[param_name] = param_def.default
        return individual

    def evolve(
        self,
        evaluate_func: Callable[[dict[str, Any]], dict[str, Any]],
        generations: int = 50,
        population_size: int = 30,
    ) -> dict[str, Any]:
        """Run evolutionary optimization with minimal I/O."""
        logger.info(
            f"Starting rapid evolution: {generations} generations, {population_size} population"
        )

        # Initialize population
        self.initialize_population(population_size)

        # Track best individual
        best_individual = None
        best_fitness = float("-inf")
        generations_without_improvement = 0

        for generation in range(generations):
            # Evaluate current population
            fitness_scores = []
            for individual in self.population:
                try:
                    result = evaluate_func(individual)
                    fitness = result.get("sharpe", float("-inf"))
                    fitness_scores.append(fitness)

                    # Track best individual
                    if fitness > best_fitness:
                        best_fitness = fitness
                        best_individual = individual.copy()
                        generations_without_improvement = 0

                        # Add to best strategies
                        self._add_to_best_strategies(individual, result)

                except Exception as e:
                    logger.warning(f"Evaluation failed: {e}")
                    fitness_scores.append(float("-inf"))

            # Track performance
            self._track_performance(generation, fitness_scores)

            # Check for early stopping
            generations_without_improvement += 1
            if generations_without_improvement >= 15:  # Early stopping
                logger.info(f"Early stopping at generation {generation}")
                break

            # Selection and reproduction
            self._evolve_population(fitness_scores)

            # Log progress
            if generation % 5 == 0:
                logger.info(
                    f"Generation {generation}: Best={best_fitness:.4f}, Avg={np.mean(fitness_scores):.4f}"
                )

        return {
            "best_individual": best_individual,
            "best_fitness": best_fitness,
            "generations_completed": len(self.performance_history["generation"]),
            "final_population_size": len(self.population),
            "diverse_strategies_found": len(self.diverse_strategies),
        }

    def _add_to_best_strategies(self, individual: dict[str, Any], result: dict[str, Any]) -> None:
        """Add high-performing strategy to best strategies list."""
        strategy_data = {
            "params": individual,
            "metrics": result,
            "timestamp": datetime.now().isoformat(),
        }

        # Keep only top 20 strategies
        self.best_strategies.append(strategy_data)
        self.best_strategies.sort(key=lambda x: x["metrics"].get("sharpe", 0), reverse=True)
        self.best_strategies = self.best_strategies[:20]

        # Check for diversity
        if self._is_diverse_strategy(individual, result):
            self.diverse_strategies.append(strategy_data)
            logger.info(f"Found diverse strategy with Sharpe: {result.get('sharpe', 0):.4f}")

    def _is_diverse_strategy(self, individual: dict[str, Any], result: dict[str, Any]) -> bool:
        """Check if strategy is diverse from existing ones."""
        if not self.diverse_strategies:
            return True

        # Simple diversity check based on parameter distance
        for existing in self.diverse_strategies:
            distance = self._calculate_distance(individual, existing["params"])
            if distance < 0.2:  # Too similar
                return False
        return True

    def _calculate_distance(self, params1: dict[str, Any], params2: dict[str, Any]) -> float:
        """Calculate normalized distance between parameter sets."""
        distances = []
        for key in set(params1.keys()) & set(params2.keys()):
            val1 = params1[key]
            val2 = params2[key]

            if isinstance(val1, int | float) and isinstance(val2, int | float):
                # Normalize by parameter range
                param_def = self.strategy_config.parameters.get(key)
                if (
                    param_def
                    and param_def.min_value is not None
                    and param_def.max_value is not None
                ):
                    range_size = param_def.max_value - param_def.min_value
                    if range_size > 0:
                        normalized_dist = abs(val1 - val2) / range_size
                        distances.append(normalized_dist)
            elif isinstance(val1, bool) and isinstance(val2, bool):
                distances.append(1.0 if val1 != val2 else 0.0)

        return np.mean(distances) if distances else 0.0

    def _track_performance(self, generation: int, fitness_scores: list[float]) -> None:
        """Track performance metrics for this generation."""
        self.performance_history["generation"].append(generation)
        self.performance_history["best_sharpe"].append(max(fitness_scores))
        self.performance_history["avg_sharpe"].append(np.mean(fitness_scores))

        # Calculate diversity score
        diversity = self._calculate_population_diversity()
        self.performance_history["diversity_score"].append(diversity)

    def _calculate_population_diversity(self) -> float:
        """Calculate diversity of current population."""
        if len(self.population) < 2:
            return 0.0

        distances = []
        for i in range(len(self.population)):
            for j in range(i + 1, len(self.population)):
                distance = self._calculate_distance(self.population[i], self.population[j])
                distances.append(distance)

        return np.mean(distances) if distances else 0.0

    def _evolve_population(self, fitness_scores: list[float]) -> None:
        """Evolve population using tournament selection, crossover, and mutation."""
        new_population = []

        # Elitism: keep best individuals
        elite_size = max(2, len(self.population) // 10)
        elite_indices = np.argsort(fitness_scores)[-elite_size:]
        for idx in elite_indices:
            new_population.append(self.population[idx].copy())

        # Generate rest of population
        while len(new_population) < len(self.population):
            # Tournament selection
            parent1 = self._tournament_selection(fitness_scores)
            parent2 = self._tournament_selection(fitness_scores)

            # Crossover
            if np.random.random() < 0.8:  # Crossover rate
                child = self._crossover(parent1, parent2)
            else:
                child = parent1.copy()

            # Mutation
            if np.random.random() < 0.2:  # Mutation rate
                child = self._mutate(child)

            new_population.append(child)

        self.population = new_population

    def _tournament_selection(
        self, fitness_scores: list[float], tournament_size: int = 3
    ) -> dict[str, Any]:
        """Select individual using tournament selection."""
        tournament_indices = np.random.choice(len(self.population), tournament_size, replace=False)
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmax(tournament_fitness)]
        return self.population[winner_idx]

    def _crossover(self, parent1: dict[str, Any], parent2: dict[str, Any]) -> dict[str, Any]:
        """Perform crossover between two parents."""
        child = {}
        for key in parent1.keys():
            if np.random.random() < 0.5:
                child[key] = parent1[key]
            else:
                child[key] = parent2[key]
        return child

    def _mutate(self, individual: dict[str, Any]) -> dict[str, Any]:
        """Mutate an individual."""
        mutated = individual.copy()
        for param_name, param_def in self.strategy_config.parameters.items():
            if np.random.random() < 0.1:  # 10% chance to mutate each parameter
                if param_def.type == "int":
                    min_val = param_def.min_value or 0
                    max_val = param_def.max_value or 100
                    mutated[param_name] = np.random.randint(min_val, max_val + 1)
                elif param_def.type == "float":
                    min_val = param_def.min_value or 0.0
                    max_val = param_def.max_value or 1.0
                    mutated[param_name] = np.random.uniform(min_val, max_val)
                elif param_def.type == "bool":
                    mutated[param_name] = not mutated[param_name]
        return mutated

    def get_summary(self) -> dict[str, Any]:
        """Get optimization summary without writing files."""
        if not self.best_strategies:
            return {"error": "No strategies found"}

        # Analyze best strategies
        sharpes = [s["metrics"].get("sharpe", 0) for s in self.best_strategies]
        cagrs = [s["metrics"].get("cagr", 0) for s in self.best_strategies]
        drawdowns = [s["metrics"].get("max_drawdown", 0) for s in self.best_strategies]

        # Identify strategy types
        strategy_types = self._identify_strategy_types()

        return {
            "total_strategies_evaluated": (
                len(self.performance_history["generation"]) * len(self.population)
            ),
            "generations_completed": len(self.performance_history["generation"]),
            "best_strategies_found": len(self.best_strategies),
            "diverse_strategies_found": len(self.diverse_strategies),
            "performance_summary": {
                "best_sharpe": max(sharpes),
                "avg_sharpe": np.mean(sharpes),
                "best_cagr": max(cagrs),
                "best_drawdown": min(drawdowns),
            },
            "strategy_types": strategy_types,
            "top_strategies": self.best_strategies[:5],  # Top 5 strategies
            "diverse_strategies": self.diverse_strategies[:5],  # Top 5 diverse strategies
        }

    def _identify_strategy_types(self) -> dict[str, int]:
        """Identify different types of strategies."""
        types = {
            "conservative": 0,
            "aggressive": 0,
            "short_term": 0,
            "long_term": 0,
            "high_frequency": 0,
        }

        for strategy in self.best_strategies:
            params = strategy["params"]

            # Conservative (low risk, high confirmation)
            if params.get("risk_pct", 0) < 0.5 and params.get("entry_confirm", 0) > 2:
                types["conservative"] += 1

            # Aggressive (high risk, low confirmation)
            if params.get("risk_pct", 0) > 2.0 and params.get("entry_confirm", 0) <= 1:
                types["aggressive"] += 1

            # Short-term (short lookback periods)
            if params.get("donchian_lookback", 0) < 50:
                types["short_term"] += 1

            # Long-term (long lookback periods)
            if params.get("donchian_lookback", 0) > 200:
                types["long_term"] += 1

            # High-frequency (no cooldown, no confirmation)
            if params.get("cooldown", 0) == 0 and params.get("entry_confirm", 0) == 0:
                types["high_frequency"] += 1

        return types

    def print_summary(self) -> None:
        """Print a human-readable summary to console."""
        summary = self.get_summary()

        print("\n" + "=" * 60)
        print("RAPID EVOLUTION SUMMARY")
        print("=" * 60)
        print(f"Total evaluations: {summary['total_strategies_evaluated']}")
        print(f"Generations completed: {summary['generations_completed']}")
        print(f"Best strategies found: {summary['best_strategies_found']}")
        print(f"Diverse strategies found: {summary['diverse_strategies_found']}")

        perf = summary["performance_summary"]
        print("\nPerformance:")
        print(f"  Best Sharpe: {perf['best_sharpe']:.4f}")
        print(f"  Average Sharpe: {perf['avg_sharpe']:.4f}")
        print(f"  Best CAGR: {perf['best_cagr']:.4f}")
        print(f"  Best Max DD: {perf['best_drawdown']:.4f}")

        types = summary["strategy_types"]
        print("\nStrategy Types:")
        for strategy_type, count in types.items():
            if count > 0:
                print(f"  {strategy_type.title()}: {count}")

        if summary["top_strategies"]:
            print("\nTop Strategy:")
            top = summary["top_strategies"][0]
            print(f"  Sharpe: {top['metrics'].get('sharpe', 0):.4f}")
            print(f"  CAGR: {top['metrics'].get('cagr', 0):.4f}")
            print(f"  Max DD: {top['metrics'].get('max_drawdown', 0):.4f}")
            print(f"  Trades: {top['metrics'].get('n_trades', 0)}")
            print(
                f"  Key params: donchian={top['params'].get('donchian_lookback')}, "
                f"atr_k={top['params'].get('atr_k'):.2f}, risk={top['params'].get('risk_pct'):.2f}"
            )

        print("=" * 60)
