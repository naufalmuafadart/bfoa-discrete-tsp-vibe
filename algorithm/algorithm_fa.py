import numpy as np
from random import random, randint
import math
from algorithm.algorithm import Algorithm


class Firefly:
    """Represents a firefly in the Firefly Algorithm."""

    def __init__(self, route, fitness):
        """
        Initialize a firefly.

        Args:
            route (list): A permutation representing the TSP route.
            fitness (float): The fitness value of the route (total distance), which corresponds to light intensity.
        """
        self.route = route
        self.fitness = fitness

    def copy(self):
        """Create a deep copy of the firefly."""
        return Firefly(self.route.copy(), self.fitness)


class FA(Algorithm):
    """Firefly Algorithm for solving TSP"""

    def __init__(self, dataset_name, population_size=25, max_iterations=20, alpha=0.7, gamma=1.0):
        """
        Initialize the Firefly Algorithm.

        Args:
            dataset_name (str): Name of the dataset.
            population_size (int): Size of the firefly population.
            max_iterations (int): Maximum number of iterations.
            alpha (float): Randomness strength.
            gamma (float): Light absorption coefficient.
        """
        super().__init__(dataset_name)
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.alpha = alpha  # Randomness factor
        self.gamma = gamma  # Light absorption coefficient

        self.population = []
        self.global_best_solution = None

    def initialize_population(self):
        """Initialize a random population of fireflies."""
        self.population = []
        for _ in range(self.population_size):
            route = np.random.permutation(self.city_count)
            fitness = self.fitness_function(route)
            self.population.append(Firefly(route, fitness))

        # Determine the initial best solution
        self.population.sort(key=lambda x: x.fitness)
        self.global_best_solution = self.population[0].copy()

    def _crossover(self, p1_route, p2_route):
        """Perform order crossover (OX1) to move a firefly towards a brighter one."""
        p1, p2 = p1_route.tolist(), p2_route.tolist()
        size = len(p1)
        child = [None] * size

        start, end = sorted([randint(0, size - 1), randint(0, size - 1)])
        if start == end:  # Ensure the slice has at least one element
            end = (start + 1) % size
            if end < start:
                start, end = end, start

        child[start:end + 1] = p1[start:end + 1]

        p2_filtered = [item for item in p2 if item not in child]

        p2_pointer = 0
        for i in range(size):
            if child[i] is None:
                child[i] = p2_filtered[p2_pointer]
                p2_pointer += 1

        return np.array(child)

    def _mutate(self, route):
        """Perform 2-opt mutation for random movement."""
        mutated_route = route.copy()
        i, j = sorted(np.random.choice(len(mutated_route), 2, replace=False))
        mutated_route[i:j + 1] = mutated_route[i:j + 1][::-1]
        return mutated_route

    def run(self):
        """Execute the Firefly Algorithm."""
        self.initialize_population()

        for t in range(self.max_iterations):
            # Sort fireflies by fitness (lower is better/brighter)
            self.population.sort(key=lambda x: x.fitness)

            # The best firefly is carried over to the next generation (elitism)
            if self.population[0].fitness < self.global_best_solution.fitness:
                self.global_best_solution = self.population[0].copy()

            new_population = [self.global_best_solution.copy()]

            # For every other firefly, attempt to move it towards a brighter one
            for i in range(1, self.population_size):
                firefly_i = self.population[i]

                # A simplified approach where fireflies are attracted to the current best
                brighter_firefly = self.population[0]

                # Move firefly 'i' towards the brighter firefly
                new_route = self._crossover(firefly_i.route, brighter_firefly.route)

                # Add a random component to the movement
                if random() < self.alpha:
                    new_route = self._mutate(new_route)

                new_fitness = self.fitness_function(new_route)

                # If the move is beneficial, update the firefly
                if new_fitness < firefly_i.fitness:
                    new_population.append(Firefly(new_route, new_fitness))
                else:
                    new_population.append(firefly_i.copy())

            self.population = new_population

        return [self.global_best_solution.route], self.global_best_solution.fitness