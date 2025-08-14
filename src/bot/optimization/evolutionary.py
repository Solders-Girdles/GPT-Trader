"""
Evolutionary optimization implementation using genetic algorithms.
"""

from __future__ import annotations

import logging
import random
from collections.abc import Callable
from typing import Any

import numpy as np
from tqdm import tqdm

from .config import OptimizationConfig

logger = logging.getLogger(__name__)


class Individual:
    """Represents an individual in the evolutionary population."""

    def __init__(self, parameters: dict[str, Any], fitness: float = float("-inf")) -> None:
        self.parameters = parameters
        self.fitness = fitness

    def __repr__(self) -> str:
        return f"Individual(fitness={self.fitness:.4f}, params={self.parameters})"


class EvolutionaryOptimizer:
    """Evolutionary optimizer using genetic algorithms."""

    def __init__(self, config: OptimizationConfig) -> None:
        self.config = config
        self.population: list[Individual] = []
        self.generation_history: list[dict[str, Any]] = []

        # Set random seed for reproducibility
        random.seed(42)
        np.random.seed(42)

    def optimize(
        self,
        evaluate_func: Callable[[dict[str, Any]], dict[str, Any]],
        initial_population: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Run evolutionary optimization.

        Args:
            evaluate_func: Function that evaluates a parameter combination
            initial_population: Initial population (optional)

        Returns:
            List of evaluation results from all generations
        """
        logger.info("Starting evolutionary optimization")

        # Initialize population
        self._initialize_population(initial_population)

        # Evaluate initial population
        self._evaluate_population(evaluate_func)

        # Evolution loop
        best_fitness_history = []
        avg_fitness_history = []

        for generation in tqdm(range(self.config.generations), desc="Evolution"):
            # Record generation statistics
            fitnesses = [ind.fitness for ind in self.population]
            best_fitness = max(fitnesses)
            avg_fitness = sum(fitnesses) / len(fitnesses)

            best_fitness_history.append(best_fitness)
            avg_fitness_history.append(avg_fitness)

            # Log progress
            logger.info(f"Generation {generation}: Best={best_fitness:.4f}, Avg={avg_fitness:.4f}")

            # Check early stopping
            if self._should_stop_early(generation, best_fitness_history):
                logger.info(f"Early stopping at generation {generation}")
                break

            # Create next generation
            self._evolve_population()

            # Evaluate new population
            self._evaluate_population(evaluate_func)

            # Record generation history
            self.generation_history.append(
                {
                    "generation": generation,
                    "best_fitness": best_fitness,
                    "avg_fitness": avg_fitness,
                    "population_size": len(self.population),
                    "best_individual": max(self.population, key=lambda x: x.fitness).parameters,
                }
            )

        # Return all evaluation results
        return self._get_all_results()

    def _initialize_population(
        self, initial_population: list[dict[str, Any]] | None = None
    ) -> None:
        """Initialize the population."""
        population_size = self.config.population_size

        if initial_population:
            # Use provided initial population
            self.population = [
                Individual(params) for params in initial_population[:population_size]
            ]

            # Fill remaining slots with random individuals
            while len(self.population) < population_size:
                individual = self._create_random_individual()
                self.population.append(individual)
        else:
            # Create entirely random population
            self.population = [self._create_random_individual() for _ in range(population_size)]

        logger.info(f"Initialized population with {len(self.population)} individuals")

    def _create_random_individual(self) -> Individual:
        """Create a random individual."""
        parameters = {}

        for param_name, param_def in self.config.parameter_space.strategy.parameters.items():
            if param_def.type == "int":
                min_val = param_def.min_value or 0
                max_val = param_def.max_value or 100
                parameters[param_name] = random.randint(min_val, max_val)
            elif param_def.type == "float":
                min_val = param_def.min_value or 0.0
                max_val = param_def.max_value or 1.0
                parameters[param_name] = random.uniform(min_val, max_val)
            elif param_def.type == "bool":
                parameters[param_name] = random.choice([True, False])
            elif param_def.type == "str" and param_def.choices:
                parameters[param_name] = random.choice(param_def.choices)
            else:
                # Use default value
                parameters[param_name] = param_def.default

        return Individual(parameters)

    def _evaluate_population(
        self, evaluate_func: Callable[[dict[str, Any]], dict[str, Any]]
    ) -> None:
        """Evaluate all individuals in the population."""
        for individual in self.population:
            if individual.fitness == float("-inf"):  # Not yet evaluated
                try:
                    result = evaluate_func(individual.parameters)
                    individual.fitness = result.get("sharpe", float("-inf"))
                except Exception as e:
                    logger.warning(f"Failed to evaluate individual: {e}")
                    individual.fitness = float("-inf")

    def _evolve_population(self) -> None:
        """Create the next generation using genetic operators."""
        new_population = []

        # Elitism: Keep best individuals
        elite_size = self.config.elite_size
        sorted_population = sorted(self.population, key=lambda x: x.fitness, reverse=True)
        new_population.extend(sorted_population[:elite_size])

        # Generate offspring through selection, crossover, and mutation
        while len(new_population) < self.config.population_size:
            # Selection
            parent1 = self._tournament_selection()
            parent2 = self._tournament_selection()

            # Crossover
            if random.random() < self.config.crossover_rate:
                child_params = self._crossover(parent1.parameters, parent2.parameters)
            else:
                child_params = parent1.parameters.copy()

            # Mutation
            child_params = self._mutate(child_params)

            # Create child individual
            child = Individual(child_params)
            new_population.append(child)

        self.population = new_population

    def _tournament_selection(self, tournament_size: int = 3) -> Individual:
        """Tournament selection."""
        tournament = random.sample(self.population, min(tournament_size, len(self.population)))
        return max(tournament, key=lambda x: x.fitness)

    def _crossover(
        self, parent1_params: dict[str, Any], parent2_params: dict[str, Any]
    ) -> dict[str, Any]:
        """Perform crossover between two parents."""
        child_params = {}

        # Single-point crossover
        param_names = list(parent1_params.keys())
        crossover_point = random.randint(1, len(param_names) - 1)

        for i, param_name in enumerate(param_names):
            if i < crossover_point:
                child_params[param_name] = parent1_params[param_name]
            else:
                child_params[param_name] = parent2_params[param_name]

        return child_params

    def _mutate(self, parameters: dict[str, Any]) -> dict[str, Any]:
        """Apply mutation to parameters."""
        mutated_params = parameters.copy()

        for param_name, param_def in self.config.parameter_space.strategy.parameters.items():
            if random.random() < self.config.mutation_rate:
                if param_def.type == "int":
                    min_val = param_def.min_value or 0
                    max_val = param_def.max_value or 100
                    mutation = random.randint(-5, 5)
                    mutated_params[param_name] = max(
                        min_val, min(max_val, mutated_params[param_name] + mutation)
                    )
                elif param_def.type == "float":
                    min_val = param_def.min_value or 0.0
                    max_val = param_def.max_value or 1.0
                    mutation = random.uniform(-0.1, 0.1)
                    mutated_params[param_name] = max(
                        min_val, min(max_val, mutated_params[param_name] + mutation)
                    )
                elif param_def.type == "bool":
                    mutated_params[param_name] = not mutated_params[param_name]
                elif param_def.type == "str" and param_def.choices:
                    mutated_params[param_name] = random.choice(param_def.choices)

        return mutated_params

    def _should_stop_early(self, generation: int, fitness_history: list[float]) -> bool:
        """Check if evolution should stop early."""
        if not self.config.early_stopping or len(fitness_history) < self.config.patience:
            return False

        # Check if there's been no improvement for patience generations
        recent_fitnesses = fitness_history[-self.config.patience :]
        best_recent = max(recent_fitnesses)
        best_overall = max(fitness_history)

        return (best_overall - best_recent) < self.config.min_improvement

    def _get_all_results(self) -> list[dict[str, Any]]:
        """Get all evaluation results from the optimization."""
        results = []

        # Add current population results
        for individual in self.population:
            results.append(
                {
                    "params": individual.parameters,
                    "sharpe": individual.fitness,
                    "generation": "final",
                }
            )

        return results

    def get_best_individual(self) -> Individual | None:
        """Get the best individual from the population."""
        if not self.population:
            return None
        return max(self.population, key=lambda x: x.fitness)

    def get_generation_history(self) -> list[dict[str, Any]]:
        """Get the history of all generations."""
        return self.generation_history
