import numpy as np
from random import random, randint
from algorithm.algorithm import Algorithm

class Wolf:
    """Represents a wolf in the Grey Wolf Optimization algorithm"""

    def __init__(self, route, fitness):
        """
        Initialize a wolf.

        Args:
            route (list): A permutation representing the TSP route.
            fitness (float): The fitness value of the route (total distance).
        """
        self.route = route
        self.fitness = fitness

    def copy(self):
        """Create a deep copy of the wolf."""
        return Wolf(self.route.copy(), self.fitness)


class GWO(Algorithm):
    """Grey Wolf Optimizer for solving TSP"""

    def __init__(self, dataset_name, population_size=10, max_iterations=10):
        """
        Initialize the GWO algorithm.

        Args:
            dataset_name (str): Name of the dataset.
            population_size (int): Size of the wolf population.
            max_iterations (int): Maximum number of iterations.
        """
        super().__init__(dataset_name)
        self.population_size = population_size
        self.max_iterations = max_iterations

        # Initialize population and hierarchy
        self.population = []
        self.alpha_wolf = None
        self.beta_wolf = None
        self.delta_wolf = None

    def initialize_population(self):
        """Initialize random wolf population and hierarchy"""
        self.population = []
        for _ in range(self.population_size):
            route = np.random.permutation(self.city_count)
            fitness = self.fitness_function(route)
            wolf = Wolf(route, fitness)
            self.population.append(wolf)

        self.update_hierarchy()

    def update_hierarchy(self):
        """Sort the population and identify alpha, beta, and delta wolves"""
        self.population.sort(key=lambda x: x.fitness)
        self.alpha_wolf = self.population[0].copy()
        self.beta_wolf = self.population[1].copy()
        self.delta_wolf = self.population[2].copy()

    def _crossover(self, p1, p2):
        """Perform order crossover (OX1) between two parent routes."""
        p1, p2 = p1.tolist(), p2.tolist()
        size = len(p1)
        child = [None] * size

        start, end = sorted([randint(0, size - 1), randint(0, size - 1)])
        if start == end:
            end = (start + 1) % size
            if end < start:
                start, end = end, start

        child[start:end] = p1[start:end]

        p2_filtered = [item for item in p2 if item not in child]

        p2_pointer = 0
        for i in range(size):
            if child[i] is None:
                child[i] = p2_filtered[p2_pointer]
                p2_pointer += 1

        return np.array(child)

    def _mutate(self, route):
        """Perform 2-opt mutation on a route."""
        mutated_route = route.copy()
        i, j = sorted(np.random.choice(len(mutated_route), 2, replace=False))
        mutated_route[i:j + 1] = mutated_route[i:j + 1][::-1]
        return mutated_route

    def run(self):
        """Execute the Grey Wolf Optimization algorithm"""
        self.initialize_population()

        for iteration in range(self.max_iterations):
            a = 2 * (1 - iteration / self.max_iterations)  # 'a' linearly decreases from 2 to 0

            new_population = []
            for wolf in self.population:
                # --- Exploitation vs. Exploration ---
                A = 2 * a * random() - a
                if abs(A) < 1:  # Exploitation: move towards the leaders
                    # Create three potential new routes based on the leaders
                    x1 = self._crossover(self.alpha_wolf.route, wolf.route)
                    x2 = self._crossover(self.beta_wolf.route, wolf.route)
                    x3 = self._crossover(self.delta_wolf.route, wolf.route)

                    # Select the best of the three
                    f1 = self.fitness_function(x1)
                    f2 = self.fitness_function(x2)
                    f3 = self.fitness_function(x3)

                    min_f = min(f1, f2, f3)
                    if min_f == f1:
                        new_route = x1
                    elif min_f == f2:
                        new_route = x2
                    else:
                        new_route = x3
                else:  # Exploration: search randomly
                    new_route = self._mutate(wolf.route)

                new_fitness = self.fitness_function(new_route)

                # Update wolf's position if the new one is better
                if new_fitness < wolf.fitness:
                    new_population.append(Wolf(new_route, new_fitness))
                else:
                    new_population.append(wolf)

            self.population = new_population
            self.update_hierarchy()

        return [self.alpha_wolf.route], self.alpha_wolf.fitness
