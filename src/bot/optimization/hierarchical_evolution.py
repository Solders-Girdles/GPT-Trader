"""
Hierarchical Strategy Evolution System.
Evolves strategy components separately then composes them for optimal performance.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from bot.optimization.config import OptimizationConfig, StrategyConfig

logger = logging.getLogger(__name__)


@dataclass
class ComponentParameters:
    """Parameters for a specific strategy component."""

    component_type: str  # "entry", "exit", "risk", "filter"
    parameters: dict[str, Any]
    performance_score: float = 0.0
    compatibility_score: float = 0.0


@dataclass
class StrategyComposition:
    """A complete strategy composed from evolved components."""

    entry_component: ComponentParameters
    exit_component: ComponentParameters
    risk_component: ComponentParameters
    filter_components: list[ComponentParameters]
    overall_performance: float = 0.0
    composition_score: float = 0.0


class ComponentEvolutionEngine:
    """Evolves individual strategy components."""

    def __init__(self, component_type: str, config: OptimizationConfig) -> None:
        self.component_type = component_type
        self.config = config
        self.evolved_components: list[ComponentParameters] = []
        self.component_performance_history: list[dict[str, Any]] = []

        # Component-specific parameter spaces
        self.parameter_spaces = self._define_parameter_spaces()

    def _define_parameter_spaces(self) -> dict[str, dict[str, Any]]:
        """Define parameter spaces for different component types."""
        return {
            "entry": {
                "donchian_lookback": {"min": 10, "max": 200, "type": "int"},
                "atr_period": {"min": 5, "max": 50, "type": "int"},
                "atr_k": {"min": 0.5, "max": 5.0, "type": "float"},
                "entry_confirmation_periods": {"min": 0, "max": 5, "type": "int"},
                "use_volume_filter": {"type": "bool"},
                "volume_threshold": {"min": 1.0, "max": 3.0, "type": "float"},
                "use_rsi_filter": {"type": "bool"},
                "rsi_oversold": {"min": 20.0, "max": 40.0, "type": "float"},
                "rsi_overbought": {"min": 60.0, "max": 80.0, "type": "float"},
            },
            "exit": {
                "exit_confirmation_periods": {"min": 0, "max": 5, "type": "int"},
                "cooldown_periods": {"min": 0, "max": 10, "type": "int"},
                "use_trailing_stop": {"type": "bool"},
                "trailing_stop_atr": {"min": 1.0, "max": 4.0, "type": "float"},
                "use_time_based_exit": {"type": "bool"},
                "max_hold_days": {"min": 5, "max": 100, "type": "int"},
            },
            "risk": {
                "max_risk_per_trade": {"min": 0.01, "max": 0.05, "type": "float"},
                "position_sizing_method": {
                    "type": "categorical",
                    "values": ["atr", "fixed", "kelly"],
                },
                "max_positions": {"min": 1, "max": 20, "type": "int"},
                "use_correlation_filter": {"type": "bool"},
                "correlation_threshold": {"min": 0.5, "max": 0.9, "type": "float"},
            },
            "filter": {
                "use_regime_filter": {"type": "bool"},
                "regime_lookback": {"min": 100, "max": 300, "type": "int"},
                "use_bollinger_filter": {"type": "bool"},
                "bollinger_period": {"min": 10, "max": 50, "type": "int"},
                "bollinger_std": {"min": 1.5, "max": 3.0, "type": "float"},
                "use_time_filter": {"type": "bool"},
                "day_of_week_filter": {"type": "categorical", "values": [None, 0, 1, 2, 3, 4]},
                "month_filter": {"type": "categorical", "values": [None] + list(range(1, 13))},
            },
        }

    def generate_random_component(self) -> ComponentParameters:
        """Generate random parameters for a component."""
        param_space = self.parameter_spaces.get(self.component_type, {})
        parameters = {}

        for param_name, param_config in param_space.items():
            param_type = param_config.get("type", "float")

            if param_type == "int":
                parameters[param_name] = np.random.randint(
                    param_config["min"], param_config["max"] + 1
                )
            elif param_type == "float":
                parameters[param_name] = np.random.uniform(param_config["min"], param_config["max"])
            elif param_type == "bool":
                parameters[param_name] = np.random.choice([True, False])
            elif param_type == "categorical":
                parameters[param_name] = np.random.choice(param_config["values"])

        return ComponentParameters(component_type=self.component_type, parameters=parameters)

    def evolve_component(
        self,
        evaluate_func: Callable[[ComponentParameters], float],
        generations: int = 50,
        population_size: int = 30,
    ) -> list[ComponentParameters]:
        """Evolve a component using single-objective optimization."""
        logger.info(f"Evolving {self.component_type} component...")

        # Initialize population
        population = [self.generate_random_component() for _ in range(population_size)]

        best_components = []

        for generation in range(generations):
            # Evaluate population
            for component in population:
                try:
                    component.performance_score = evaluate_func(component)
                except Exception as e:
                    logger.warning(f"Component evaluation failed: {e}")
                    component.performance_score = 0.0

            # Sort by performance
            population.sort(key=lambda x: x.performance_score, reverse=True)

            # Keep best components
            best_components = population[:5].copy()

            # Track performance
            self.component_performance_history.append(
                {
                    "generation": generation,
                    "best_score": population[0].performance_score,
                    "avg_score": np.mean([c.performance_score for c in population]),
                }
            )

            # Create next generation
            new_population = []

            # Elitism: keep best 20%
            elite_size = max(1, population_size // 5)
            new_population.extend(population[:elite_size])

            # Generate offspring
            while len(new_population) < population_size:
                # Tournament selection
                parent1 = self._tournament_selection(population)
                parent2 = self._tournament_selection(population)

                # Crossover
                child = self._crossover_components(parent1, parent2)

                # Mutation
                if np.random.random() < 0.2:
                    child = self._mutate_component(child)

                new_population.append(child)

            population = new_population

            if generation % 10 == 0:
                logger.info(
                    f"Generation {generation}: Best score = {best_components[0].performance_score:.4f}"
                )

        self.evolved_components = best_components
        return best_components

    def _tournament_selection(self, population: list[ComponentParameters]) -> ComponentParameters:
        """Tournament selection for components."""
        tournament_size = 3
        tournament = np.random.choice(population, tournament_size, replace=False)
        return max(tournament, key=lambda x: x.performance_score)

    def _crossover_components(
        self, parent1: ComponentParameters, parent2: ComponentParameters
    ) -> ComponentParameters:
        """Crossover two components."""
        child_params = {}

        for param_name in parent1.parameters:
            if np.random.random() < 0.5:
                child_params[param_name] = parent1.parameters[param_name]
            else:
                child_params[param_name] = parent2.parameters[param_name]

        return ComponentParameters(component_type=self.component_type, parameters=child_params)

    def _mutate_component(self, component: ComponentParameters) -> ComponentParameters:
        """Mutate a component."""
        param_space = self.parameter_spaces.get(self.component_type, {})
        mutated_params = component.parameters.copy()

        for param_name, param_config in param_space.items():
            if np.random.random() < 0.1:  # 10% mutation probability
                param_type = param_config.get("type", "float")

                if param_type == "int":
                    current_val = mutated_params[param_name]
                    mutated_params[param_name] = max(
                        param_config["min"],
                        min(param_config["max"], current_val + np.random.randint(-5, 6)),
                    )
                elif param_type == "float":
                    current_val = mutated_params[param_name]
                    mutated_params[param_name] = max(
                        param_config["min"],
                        min(param_config["max"], current_val + np.random.uniform(-0.2, 0.2)),
                    )
                elif param_type == "bool":
                    mutated_params[param_name] = not mutated_params[param_name]
                elif param_type == "categorical":
                    mutated_params[param_name] = np.random.choice(param_config["values"])

        return ComponentParameters(
            component_type=component.component_type, parameters=mutated_params
        )


class StrategyCompositionEngine:
    """Composes strategies from evolved components."""

    def __init__(self) -> None:
        self.composition_rules: list[dict[str, Any]] = []
        self.compatibility_matrix: dict[str, dict[str, float]] = {}
        self.composed_strategies: list[StrategyComposition] = []

    def compose_strategies(
        self,
        evolved_components: dict[str, list[ComponentParameters]],
        evaluate_composition: Callable[[StrategyComposition], float],
    ) -> list[StrategyComposition]:
        """Compose strategies from evolved components."""
        logger.info("Composing strategies from evolved components...")

        # Extract components by type
        entry_components = evolved_components.get("entry", [])
        exit_components = evolved_components.get("exit", [])
        risk_components = evolved_components.get("risk", [])
        filter_components = evolved_components.get("filter", [])

        if not all([entry_components, exit_components, risk_components]):
            logger.error("Missing required components for composition")
            return []

        # Generate compositions
        compositions = []

        # Try different combinations
        for entry_comp in entry_components[:3]:  # Top 3 entry components
            for exit_comp in exit_components[:3]:  # Top 3 exit components
                for risk_comp in risk_components[:3]:  # Top 3 risk components
                    # Create composition
                    composition = StrategyComposition(
                        entry_component=entry_comp,
                        exit_component=exit_comp,
                        risk_component=risk_comp,
                        filter_components=filter_components[:2],  # Top 2 filter components
                    )

                    # Evaluate composition
                    try:
                        composition.overall_performance = evaluate_composition(composition)
                        composition.composition_score = self._calculate_composition_score(
                            composition
                        )
                        compositions.append(composition)
                    except Exception as e:
                        logger.warning(f"Composition evaluation failed: {e}")

        # Sort by performance
        compositions.sort(key=lambda x: x.overall_performance, reverse=True)

        self.composed_strategies = compositions[:10]  # Keep top 10
        return self.composed_strategies

    def _calculate_composition_score(self, composition: StrategyComposition) -> float:
        """Calculate composition score based on component compatibility."""
        score = 0.0

        # Base score from component performance
        score += composition.entry_component.performance_score * 0.3
        score += composition.exit_component.performance_score * 0.3
        score += composition.risk_component.performance_score * 0.2

        for filter_comp in composition.filter_components:
            score += filter_comp.performance_score * 0.1

        # Compatibility bonus
        compatibility_bonus = self._calculate_compatibility_bonus(composition)
        score += compatibility_bonus

        return score

    def _calculate_compatibility_bonus(self, composition: StrategyComposition) -> float:
        """Calculate compatibility bonus between components."""
        bonus = 0.0

        # Entry-Exit compatibility
        entry_params = composition.entry_component.parameters
        exit_params = composition.exit_component.parameters

        # Check for logical conflicts
        if (
            entry_params.get("entry_confirmation_periods", 0) > 0
            and exit_params.get("exit_confirmation_periods", 0) > 0
        ):
            bonus += 0.1  # Both use confirmation periods

        # Risk-Entry compatibility
        risk_params = composition.risk_component.parameters
        if entry_params.get("use_volume_filter", False) and risk_params.get(
            "use_correlation_filter", False
        ):
            bonus += 0.05  # Both use filtering

        return bonus


class HierarchicalEvolutionEngine:
    """Main hierarchical evolution engine."""

    def __init__(self, config: OptimizationConfig, strategy_config: StrategyConfig) -> None:
        self.config = config
        self.strategy_config = strategy_config

        # Component evolution engines
        self.component_engines = {
            "entry": ComponentEvolutionEngine("entry", config),
            "exit": ComponentEvolutionEngine("exit", config),
            "risk": ComponentEvolutionEngine("risk", config),
            "filter": ComponentEvolutionEngine("filter", config),
        }

        # Composition engine
        self.composition_engine = StrategyCompositionEngine()

        # Results storage
        self.evolved_components: dict[str, list[ComponentParameters]] = {}
        self.composed_strategies: list[StrategyComposition] = []
        self.evolution_history: list[dict[str, Any]] = []

    def evolve_hierarchically(
        self,
        component_evaluators: dict[str, Callable[[ComponentParameters], float]],
        composition_evaluator: Callable[[StrategyComposition], float],
        generations: int = 50,
        population_size: int = 30,
    ) -> dict[str, Any]:
        """Run hierarchical evolution process."""
        logger.info("Starting hierarchical strategy evolution...")

        # Phase 1: Evolve components separately
        logger.info("Phase 1: Evolving individual components...")
        for component_type, evaluator in component_evaluators.items():
            if component_type in self.component_engines:
                evolved_components = self.component_engines[component_type].evolve_component(
                    evaluator, generations, population_size
                )
                self.evolved_components[component_type] = evolved_components
                logger.info(f"Evolved {len(evolved_components)} {component_type} components")

        # Phase 2: Compose strategies from evolved components
        logger.info("Phase 2: Composing strategies from evolved components...")
        self.composed_strategies = self.composition_engine.compose_strategies(
            self.evolved_components, composition_evaluator
        )

        # Generate results
        results = self._generate_results()

        logger.info(
            f"Hierarchical evolution completed. Generated {len(self.composed_strategies)} strategies."
        )
        return results

    def _generate_results(self) -> dict[str, Any]:
        """Generate comprehensive results."""
        return {
            "evolved_components": self.evolved_components,
            "composed_strategies": self.composed_strategies,
            "best_strategy": (
                max(self.composed_strategies, key=lambda x: x.overall_performance)
                if self.composed_strategies
                else None
            ),
            "component_performance": self._analyze_component_performance(),
            "composition_analysis": self._analyze_compositions(),
            "evolution_history": self.evolution_history,
        }

    def _analyze_component_performance(self) -> dict[str, Any]:
        """Analyze performance of evolved components."""
        analysis = {}

        for component_type, components in self.evolved_components.items():
            if components:
                scores = [c.performance_score for c in components]
                analysis[component_type] = {
                    "count": len(components),
                    "best_score": max(scores),
                    "avg_score": np.mean(scores),
                    "std_score": np.std(scores),
                    "top_components": [
                        {"parameters": c.parameters, "performance_score": c.performance_score}
                        for c in sorted(
                            components, key=lambda x: x.performance_score, reverse=True
                        )[:3]
                    ],
                }

        return analysis

    def _analyze_compositions(self) -> dict[str, Any]:
        """Analyze composed strategies."""
        if not self.composed_strategies:
            return {}

        performance_scores = [s.overall_performance for s in self.composed_strategies]
        composition_scores = [s.composition_score for s in self.composed_strategies]

        return {
            "total_compositions": len(self.composed_strategies),
            "best_performance": max(performance_scores),
            "avg_performance": np.mean(performance_scores),
            "best_composition_score": max(composition_scores),
            "avg_composition_score": np.mean(composition_scores),
            "top_strategies": [
                {
                    "entry_component": s.entry_component.parameters,
                    "exit_component": s.exit_component.parameters,
                    "risk_component": s.risk_component.parameters,
                    "filter_components": [f.parameters for f in s.filter_components],
                    "overall_performance": s.overall_performance,
                    "composition_score": s.composition_score,
                }
                for s in sorted(
                    self.composed_strategies, key=lambda x: x.overall_performance, reverse=True
                )[:5]
            ],
        }
