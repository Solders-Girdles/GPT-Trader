"""
Enhanced evolutionary optimization with expanded parameter space and novel genetic operators.
Designed to discover surprising strategies by exploring a much wider solution space.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
from bot.optimization.config import OptimizationConfig, StrategyConfig
from bot.optimization.strategy_diversity import StrategyDiversityTracker

logger = logging.getLogger(__name__)


@dataclass
class EnhancedStrategyParams:
    """Enhanced strategy parameters with expanded feature set."""

    # Core trend parameters
    donchian_lookback: int = 55
    atr_period: int = 20
    atr_k: float = 2.0

    # Volume-based features
    volume_ma_period: int = 20
    volume_threshold: float = 1.5
    use_volume_filter: bool = True

    # Momentum features
    rsi_period: int = 14
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0
    use_rsi_filter: bool = False

    # Volatility features
    bollinger_period: int = 20
    bollinger_std: float = 2.0
    use_bollinger_filter: bool = False

    # Time-based features
    day_of_week_filter: int | None = None  # 0=Monday, 4=Friday
    month_filter: int | None = None  # 1-12
    use_time_filter: bool = False

    # Entry/Exit enhancements
    entry_confirmation_periods: int = 1
    exit_confirmation_periods: int = 1
    cooldown_periods: int = 0

    # Risk management
    max_risk_per_trade: float = 0.02
    position_sizing_method: str = "atr"  # "atr", "fixed", "kelly"

    # Advanced features
    use_regime_filter: bool = False
    regime_lookback: int = 200
    use_correlation_filter: bool = False
    correlation_threshold: float = 0.7
    correlation_lookback: int = 60


class EnhancedEvolutionEngine:
    """Enhanced evolutionary optimization with expanded parameter space and novel operators."""

    def __init__(self, config: OptimizationConfig, strategy_config: StrategyConfig) -> None:
        self.config = config
        self.strategy_config = strategy_config
        self.population: list[dict[str, Any]] = []
        self.generation_history: list[dict[str, Any]] = []
        self.best_strategies: list[dict[str, Any]] = []
        self.diverse_strategies: list[dict[str, Any]] = []

        # Enhanced diversity tracking
        from pathlib import Path

        self.diversity_tracker = StrategyDiversityTracker(
            output_dir=Path(config.output_dir), strategy_config=strategy_config
        )

        # Novelty archive for surprising strategies
        self.novelty_archive: list[dict[str, Any]] = []
        self.novelty_threshold = 0.3

        # Performance tracking
        self.performance_history = {
            "best_sharpe": [],
            "avg_sharpe": [],
            "diversity_score": [],
            "novelty_score": [],
            "generation": [],
        }

        # Adaptive mutation rates
        self.mutation_rates = {
            "exploration": 0.3,  # High mutation for exploration
            "exploitation": 0.1,  # Low mutation for exploitation
            "novelty": 0.5,  # Very high mutation for novelty search
        }

        # Strategy type tracking
        self.strategy_types = {
            "trend_following": 0,
            "mean_reversion": 0,
            "momentum": 0,
            "volatility": 0,
            "multi_timeframe": 0,
            "hybrid": 0,
        }
        # Optional surrogate model (scaffolding for model-guided search in research mode)
        self.surrogate_model = None

    def initialize_population(self, size: int) -> None:
        """Initialize diverse population with expanded parameter space."""
        self.population = []

        # Create different strategy archetypes
        archetypes = self._generate_strategy_archetypes()

        for i in range(size):
            if i < len(archetypes):
                # Use predefined archetypes for diversity
                individual = archetypes[i]
            else:
                # Generate random individual with expanded parameters
                individual = self._generate_enhanced_individual()

            self.population.append(individual)

        logger.info(f"Initialized enhanced population of {size} individuals")

    def _generate_strategy_archetypes(self) -> list[dict[str, Any]]:
        """Generate diverse strategy archetypes to seed the population."""
        archetypes = []

        # 1. Trend Following
        archetypes.append(
            {
                "donchian_lookback": 55,
                "atr_period": 20,
                "atr_k": 2.0,
                "volume_ma_period": 20,
                "volume_threshold": 1.5,
                "use_volume_filter": True,
                "rsi_period": 14,
                "rsi_oversold": 30.0,
                "rsi_overbought": 70.0,
                "use_rsi_filter": False,
                "bollinger_period": 20,
                "bollinger_std": 2.0,
                "use_bollinger_filter": False,
                "day_of_week_filter": None,
                "month_filter": None,
                "use_time_filter": False,
                "entry_confirmation_periods": 1,
                "exit_confirmation_periods": 1,
                "cooldown_periods": 0,
                "max_risk_per_trade": 0.02,
                "position_sizing_method": "atr",
                "use_regime_filter": False,
                "regime_lookback": 200,
                "use_correlation_filter": False,
                "correlation_threshold": 0.7,
                "correlation_lookback": 60,
            }
        )

        # 2. Mean Reversion
        archetypes.append(
            {
                "donchian_lookback": 20,
                "atr_period": 14,
                "atr_k": 1.5,
                "volume_ma_period": 10,
                "volume_threshold": 2.0,
                "use_volume_filter": True,
                "rsi_period": 14,
                "rsi_oversold": 25.0,
                "rsi_overbought": 75.0,
                "use_rsi_filter": True,
                "bollinger_period": 20,
                "bollinger_std": 2.0,
                "use_bollinger_filter": True,
                "day_of_week_filter": None,
                "month_filter": None,
                "use_time_filter": False,
                "entry_confirmation_periods": 2,
                "exit_confirmation_periods": 1,
                "cooldown_periods": 5,
                "max_risk_per_trade": 0.015,
                "position_sizing_method": "fixed",
                "use_regime_filter": True,
                "regime_lookback": 100,
                "use_correlation_filter": False,
                "correlation_threshold": 0.7,
                "correlation_lookback": 60,
            }
        )

        # 3. Momentum
        archetypes.append(
            {
                "donchian_lookback": 10,
                "atr_period": 10,
                "atr_k": 1.0,
                "volume_ma_period": 5,
                "volume_threshold": 1.2,
                "use_volume_filter": True,
                "rsi_period": 7,
                "rsi_oversold": 20.0,
                "rsi_overbought": 80.0,
                "use_rsi_filter": False,
                "bollinger_period": 10,
                "bollinger_std": 1.5,
                "use_bollinger_filter": False,
                "day_of_week_filter": None,
                "month_filter": None,
                "use_time_filter": False,
                "entry_confirmation_periods": 0,
                "exit_confirmation_periods": 0,
                "cooldown_periods": 0,
                "max_risk_per_trade": 0.03,
                "position_sizing_method": "kelly",
                "use_regime_filter": False,
                "regime_lookback": 200,
                "use_correlation_filter": True,
                "correlation_threshold": 0.8,
                "correlation_lookback": 30,
            }
        )

        # 4. Volatility Breakout
        archetypes.append(
            {
                "donchian_lookback": 100,
                "atr_period": 30,
                "atr_k": 3.0,
                "volume_ma_period": 30,
                "volume_threshold": 2.5,
                "use_volume_filter": True,
                "rsi_period": 21,
                "rsi_oversold": 35.0,
                "rsi_overbought": 65.0,
                "use_rsi_filter": False,
                "bollinger_period": 30,
                "bollinger_std": 3.0,
                "use_bollinger_filter": True,
                "day_of_week_filter": None,
                "month_filter": None,
                "use_time_filter": False,
                "entry_confirmation_periods": 3,
                "exit_confirmation_periods": 2,
                "cooldown_periods": 10,
                "max_risk_per_trade": 0.01,
                "position_sizing_method": "atr",
                "use_regime_filter": True,
                "regime_lookback": 300,
                "use_correlation_filter": False,
                "correlation_threshold": 0.7,
                "correlation_lookback": 60,
            }
        )

        return archetypes

    def _generate_enhanced_individual(self) -> dict[str, Any]:
        """Generate a random individual with expanded parameter space."""
        individual = {}

        # Core trend parameters
        individual["donchian_lookback"] = np.random.randint(5, 200)
        individual["atr_period"] = np.random.randint(5, 50)
        individual["atr_k"] = np.random.uniform(0.5, 5.0)

        # Volume features
        individual["volume_ma_period"] = np.random.randint(5, 50)
        individual["volume_threshold"] = np.random.uniform(1.0, 5.0)
        individual["use_volume_filter"] = np.random.choice([True, False])

        # Momentum features
        individual["rsi_period"] = np.random.randint(5, 30)
        individual["rsi_oversold"] = np.random.uniform(20.0, 40.0)
        individual["rsi_overbought"] = np.random.uniform(60.0, 80.0)
        individual["use_rsi_filter"] = np.random.choice([True, False])

        # Volatility features
        individual["bollinger_period"] = np.random.randint(10, 50)
        individual["bollinger_std"] = np.random.uniform(1.0, 4.0)
        individual["use_bollinger_filter"] = np.random.choice([True, False])

        # Time filters
        individual["day_of_week_filter"] = np.random.choice([None] + list(range(5)))
        individual["month_filter"] = np.random.choice([None] + list(range(1, 13)))
        individual["use_time_filter"] = np.random.choice([True, False])

        # Entry/Exit
        individual["entry_confirmation_periods"] = np.random.randint(0, 5)
        individual["exit_confirmation_periods"] = np.random.randint(0, 5)
        individual["cooldown_periods"] = np.random.randint(0, 20)

        # Risk management
        individual["max_risk_per_trade"] = np.random.uniform(0.005, 0.05)
        individual["position_sizing_method"] = np.random.choice(["atr", "fixed", "kelly"])

        # Advanced features
        individual["use_regime_filter"] = np.random.choice([True, False])
        individual["regime_lookback"] = np.random.randint(50, 500)
        individual["use_correlation_filter"] = np.random.choice([True, False])
        individual["correlation_threshold"] = np.random.uniform(0.5, 0.9)
        individual["correlation_lookback"] = np.random.randint(20, 120)

        return individual

    def evolve(
        self,
        evaluate_func: Callable[[dict[str, Any]], dict[str, Any]],
        generations: int = 100,
        population_size: int = 50,
    ) -> dict[str, Any]:
        """Run enhanced evolutionary optimization."""
        logger.info(
            f"Starting enhanced evolution: {generations} generations, {population_size} population"
        )

        # Initialize population
        self.initialize_population(population_size)

        # Track best individual
        best_individual = None
        best_fitness = float("-inf")
        generations_without_improvement = 0

        # Adaptive parameters
        current_phase = "exploration"
        novelty_weight = 0.3

        for generation in range(generations):
            # Optional: pre-rank population with surrogate (no-op if not set)
            if self.surrogate_model is not None:
                try:
                    self._pre_rank_with_surrogate()
                except Exception:
                    pass
            # Evaluate current population
            fitness_scores = []
            novelty_scores = []

            for individual in self.population:
                try:
                    result = evaluate_func(individual)
                    fitness = result.get("sharpe", float("-inf"))
                    fitness_scores.append(fitness)

                    # Calculate novelty score
                    novelty = self._calculate_novelty_score(individual)
                    novelty_scores.append(novelty)

                    # Combined fitness (fitness + novelty)
                    combined_fitness = fitness + novelty_weight * novelty

                    # Track best individual
                    if combined_fitness > best_fitness:
                        best_fitness = combined_fitness
                        best_individual = individual.copy()
                        generations_without_improvement = 0

                        # Add to best strategies
                        self._add_to_best_strategies(individual, result)

                        # Check for novelty
                        if novelty > self.novelty_threshold:
                            self._add_to_novelty_archive(individual, result, novelty)

                except Exception as e:
                    logger.warning(f"Evaluation failed: {e}")
                    fitness_scores.append(float("-inf"))
                    novelty_scores.append(0.0)

            # Track performance
            self._track_enhanced_performance(generation, fitness_scores, novelty_scores)

            # Adaptive phase switching
            if generation > 20:
                diversity = self._calculate_population_diversity()
                if diversity < 0.2 and current_phase == "exploration":
                    current_phase = "exploitation"
                    novelty_weight = 0.1
                    logger.info(f"Switching to exploitation phase at generation {generation}")
                elif diversity > 0.4 and current_phase == "exploitation":
                    current_phase = "exploration"
                    novelty_weight = 0.3
                    logger.info(f"Switching to exploration phase at generation {generation}")

            # Check for early stopping
            generations_without_improvement += 1
            if generations_without_improvement >= 25:
                logger.info(f"Early stopping at generation {generation}")
                break

            # Enhanced selection and reproduction
            self._evolve_population_enhanced(fitness_scores, novelty_scores, current_phase)

            # Log progress
            if generation % 10 == 0:
                avg_fitness = np.mean([f for f in fitness_scores if f > float("-inf")])
                avg_novelty = np.mean(novelty_scores)
                diversity = self._calculate_population_diversity()
                logger.info(
                    f"Generation {generation}: Best={best_fitness:.4f}, Avg={avg_fitness:.4f}, "
                    f"Novelty={avg_novelty:.4f}, Diversity={diversity:.4f}"
                )

        return {
            "best_individual": best_individual,
            "best_fitness": best_fitness,
            "generations_completed": len(self.performance_history["generation"]),
            "final_population_size": len(self.population),
            "diverse_strategies_found": len(self.diverse_strategies),
            "novel_strategies_found": len(self.novelty_archive),
            "strategy_types": self._identify_strategy_types(),
        }

    # --- Surrogate hooks (scaffolding) ---
    def set_surrogate(self, model: Any) -> None:
        """Attach a surrogate model for pre-ranking/filtering (research mode only)."""
        self.surrogate_model = model

    def _pre_rank_with_surrogate(self) -> None:
        """Optional pre-ranking step using a surrogate model.

        Implementations can score individuals based on derived features and reorder or prune.
        No-ops by default to remain backward-compatible.
        """
        return

    def _calculate_novelty_score(self, individual: dict[str, Any]) -> float:
        """Calculate novelty score based on distance from archive."""
        if not self.novelty_archive:
            return 1.0  # First individual is maximally novel

        distances = []
        for archived in self.novelty_archive:
            distance = self._calculate_distance(individual, archived["params"])
            distances.append(distance)

        # Novelty is the minimum distance to any archived individual
        return min(distances) if distances else 1.0

    def _add_to_novelty_archive(
        self, individual: dict[str, Any], result: dict[str, Any], novelty: float
    ) -> None:
        """Add novel strategy to archive."""
        strategy_data = {
            "params": individual,
            "metrics": result,
            "novelty_score": novelty,
            "timestamp": datetime.now().isoformat(),
        }

        self.novelty_archive.append(strategy_data)
        logger.info(
            f"Found novel strategy with novelty: {novelty:.4f}, Sharpe: {result.get('sharpe', 0):.4f}"
        )

    def _evolve_population_enhanced(
        self, fitness_scores: list[float], novelty_scores: list[float], phase: str
    ) -> None:
        """Enhanced population evolution with multiple operators."""
        new_population = []

        # Elitism: keep best individuals
        elite_size = max(3, len(self.population) // 10)
        combined_scores = [
            f + 0.3 * n for f, n in zip(fitness_scores, novelty_scores, strict=False)
        ]
        elite_indices = np.argsort(combined_scores)[-elite_size:]
        for idx in elite_indices:
            new_population.append(self.population[idx].copy())

        # Generate rest of population using multiple operators
        while len(new_population) < len(self.population):
            operator = np.random.choice(
                ["crossover", "mutation", "novelty_search", "archive_injection"],
                p=[0.4, 0.3, 0.2, 0.1],
            )

            if operator == "crossover":
                parent1 = self._tournament_selection_enhanced(fitness_scores, novelty_scores)
                parent2 = self._tournament_selection_enhanced(fitness_scores, novelty_scores)
                child = self._enhanced_crossover(parent1, parent2)

            elif operator == "mutation":
                parent = self._tournament_selection_enhanced(fitness_scores, novelty_scores)
                child = self._enhanced_mutation(parent, phase)

            elif operator == "novelty_search":
                child = self._novelty_search_mutation()

            else:  # archive_injection
                child = self._inject_from_archive()

            new_population.append(child)

        self.population = new_population

    def _tournament_selection_enhanced(
        self, fitness_scores: list[float], novelty_scores: list[float], tournament_size: int = 5
    ) -> dict[str, Any]:
        """Enhanced tournament selection considering both fitness and novelty."""
        tournament_indices = np.random.choice(len(self.population), tournament_size, replace=False)
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        tournament_novelty = [novelty_scores[i] for i in tournament_indices]

        # Combined score
        combined_scores = [
            f + 0.3 * n for f, n in zip(tournament_fitness, tournament_novelty, strict=False)
        ]
        winner_idx = tournament_indices[np.argmax(combined_scores)]

        return self.population[winner_idx]

    def _enhanced_crossover(
        self, parent1: dict[str, Any], parent2: dict[str, Any]
    ) -> dict[str, Any]:
        """Enhanced crossover with different methods for different parameter types."""
        child = {}

        for key in parent1.keys():
            if key in ["position_sizing_method"]:
                # Categorical crossover
                child[key] = np.random.choice([parent1[key], parent2[key]])
            elif key in [
                "use_volume_filter",
                "use_rsi_filter",
                "use_bollinger_filter",
                "use_time_filter",
                "use_regime_filter",
                "use_correlation_filter",
            ]:
                # Boolean crossover with bias toward True
                if parent1[key] and parent2[key]:
                    child[key] = True
                elif not parent1[key] and not parent2[key]:
                    child[key] = np.random.choice([True, False], p=[0.3, 0.7])
                else:
                    child[key] = np.random.choice([True, False])
            else:
                # Numerical crossover with interpolation
                if np.random.random() < 0.5:
                    child[key] = parent1[key]
                else:
                    child[key] = parent2[key]

                    # Add small random perturbation
                    if isinstance(child[key], int | float) and child[key] is not None:
                        if isinstance(child[key], int):
                            child[key] += np.random.randint(-2, 3)
                        else:
                            child[key] *= np.random.uniform(0.9, 1.1)

        return child

    def _enhanced_mutation(self, individual: dict[str, Any], phase: str) -> dict[str, Any]:
        """Enhanced mutation with phase-dependent rates."""
        mutated = individual.copy()
        mutation_rate = self.mutation_rates[phase]

        for key, value in mutated.items():
            if np.random.random() < mutation_rate:
                if key in [
                    "donchian_lookback",
                    "atr_period",
                    "volume_ma_period",
                    "rsi_period",
                    "bollinger_period",
                    "entry_confirmation_periods",
                    "exit_confirmation_periods",
                    "cooldown_periods",
                    "regime_lookback",
                    "correlation_lookback",
                ]:
                    # Integer parameters
                    mutated[key] = max(1, value + np.random.randint(-5, 6))

                elif key in [
                    "atr_k",
                    "volume_threshold",
                    "rsi_oversold",
                    "rsi_overbought",
                    "bollinger_std",
                    "max_risk_per_trade",
                    "correlation_threshold",
                ]:
                    # Float parameters
                    mutated[key] = value * np.random.uniform(0.7, 1.3)

                elif key in [
                    "use_volume_filter",
                    "use_rsi_filter",
                    "use_bollinger_filter",
                    "use_time_filter",
                    "use_regime_filter",
                    "use_correlation_filter",
                ]:
                    # Boolean parameters
                    mutated[key] = not value

                elif key == "position_sizing_method":
                    # Categorical parameters
                    mutated[key] = np.random.choice(["atr", "fixed", "kelly"])

                elif key in ["day_of_week_filter", "month_filter"]:
                    # Optional parameters
                    if value is None:
                        mutated[key] = np.random.choice(
                            [None] + list(range(5 if "day" in key else 12))
                        )
                    else:
                        mutated[key] = None

        return mutated

    def _novelty_search_mutation(self) -> dict[str, Any]:
        """Generate novel individual by combining archived strategies."""
        if len(self.novelty_archive) < 2:
            return self._generate_enhanced_individual()

        # Select two random archived strategies
        archived1, archived2 = np.random.choice(self.novelty_archive, 2, replace=False)

        # Combine them with high mutation
        child = self._enhanced_crossover(archived1["params"], archived2["params"])
        child = self._enhanced_mutation(child, "novelty")

        return child

    def _inject_from_archive(self) -> dict[str, Any]:
        """Inject individual from novelty archive."""
        if not self.novelty_archive:
            return self._generate_enhanced_individual()

        archived = np.random.choice(self.novelty_archive)
        return archived["params"].copy()

    def _calculate_distance(
        self, individual1: dict[str, Any], individual2: dict[str, Any]
    ) -> float:
        """Calculate distance between two individuals."""
        distance = 0.0
        count = 0

        for key in individual1.keys():
            if key in individual2:
                val1, val2 = individual1[key], individual2[key]

                if isinstance(val1, bool) and isinstance(val2, bool):
                    distance += 1.0 if val1 != val2 else 0.0
                elif isinstance(val1, str) and isinstance(val2, str):
                    distance += 1.0 if val1 != val2 else 0.0
                elif isinstance(val1, int | float) and isinstance(val2, int | float):
                    if val1 is not None and val2 is not None:
                        # Normalize by typical range
                        if "lookback" in key or "period" in key:
                            distance += abs(val1 - val2) / 100.0
                        elif "threshold" in key or "std" in key:
                            distance += abs(val1 - val2) / 5.0
                        else:
                            distance += abs(val1 - val2) / 1.0
                count += 1

        return distance / count if count > 0 else 1.0

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

    def _track_enhanced_performance(
        self, generation: int, fitness_scores: list[float], novelty_scores: list[float]
    ) -> None:
        """Track enhanced performance metrics."""
        valid_fitness = [f for f in fitness_scores if f > float("-inf")]

        self.performance_history["generation"].append(generation)
        self.performance_history["best_sharpe"].append(
            max(valid_fitness) if valid_fitness else float("-inf")
        )
        self.performance_history["avg_sharpe"].append(
            np.mean(valid_fitness) if valid_fitness else float("-inf")
        )
        self.performance_history["diversity_score"].append(self._calculate_population_diversity())
        self.performance_history["novelty_score"].append(
            np.mean(novelty_scores) if novelty_scores else 0.0
        )

    def _add_to_best_strategies(self, individual: dict[str, Any], result: dict[str, Any]) -> None:
        """Add high-performing strategy to best strategies list."""
        strategy_data = {
            "params": individual,
            "metrics": result,
            "timestamp": datetime.now().isoformat(),
        }

        # Keep only top 30 strategies
        self.best_strategies.append(strategy_data)
        self.best_strategies.sort(key=lambda x: x["metrics"].get("sharpe", 0), reverse=True)
        self.best_strategies = self.best_strategies[:30]

        # Check for diversity
        if self._is_diverse_strategy(individual, result):
            self.diverse_strategies.append(strategy_data)
            logger.info(f"Found diverse strategy with Sharpe: {result.get('sharpe', 0):.4f}")

    def _is_diverse_strategy(self, individual: dict[str, Any], result: dict[str, Any]) -> bool:
        """Check if strategy is diverse from existing ones."""
        if not self.diverse_strategies:
            return True

        # Check distance to existing diverse strategies
        for existing in self.diverse_strategies:
            distance = self._calculate_distance(individual, existing["params"])
            if distance < 0.15:  # More strict diversity requirement
                return False

        return True

    def _identify_strategy_types(self) -> dict[str, int]:
        """Identify different types of strategies in the population."""
        types = {
            "trend_following": 0,
            "mean_reversion": 0,
            "momentum": 0,
            "volatility": 0,
            "multi_timeframe": 0,
            "hybrid": 0,
        }

        for strategy in self.best_strategies:
            params = strategy["params"]

            # Trend following (long lookback, no RSI filter)
            if params.get("donchian_lookback", 0) > 50 and not params.get("use_rsi_filter", False):
                types["trend_following"] += 1

            # Mean reversion (short lookback, RSI filter)
            elif params.get("donchian_lookback", 0) < 30 and params.get("use_rsi_filter", False):
                types["mean_reversion"] += 1

            # Momentum (very short lookback, no confirmation)
            elif (
                params.get("donchian_lookback", 0) < 20
                and params.get("entry_confirmation_periods", 0) == 0
            ):
                types["momentum"] += 1

            # Volatility (high ATR k, Bollinger filter)
            elif params.get("atr_k", 0) > 2.5 and params.get("use_bollinger_filter", False):
                types["volatility"] += 1

            # Multi-timeframe (regime filter, correlation filter)
            elif params.get("use_regime_filter", False) and params.get(
                "use_correlation_filter", False
            ):
                types["multi_timeframe"] += 1

            # Hybrid (multiple filters)
            else:
                types["hybrid"] += 1

        return types
