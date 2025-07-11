import numpy as np
from random import random, randint
import math
from algorithm.algorithm import Algorithm


class Whale:
    """Represents a whale in the Whale Optimization Algorithm"""

    def __init__(self, route, fitness):
        """
        Initialize a whale.

        Args:
            route (list): A permutation representing the TSP route
            fitness (float): The fitness value of the route (total distance)
        """
        self.route = route
        self.fitness = fitness

    def copy(self):
        """Create a deep copy of the whale"""
        return Whale(self.route.copy(), self.fitness)


class WOA(Algorithm):
    """Whale Optimization Algorithm for solving TSP"""

    def __init__(self, dataset_name, population_size=10, max_iterations=10, b=1.0):
        """
        Initialize the WOA algorithm.

        Args:
            dataset_name (str): Name of the dataset
            population_size (int): Size of the whale population
            max_iterations (int): Maximum number of iterations
            b (float): Spiral path shape constant
        """
        super().__init__(dataset_name)
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.b = b  # spiral shape constant

        # Initialize population and best solution
        self.population = []
        self.best_solution = None
        self.best_fitness = float('inf')

    def initialize_population(self):
        """Initialize random whale population"""
        self.population = []
        for _ in range(self.population_size):
            route = np.random.permutation(self.city_count)
            fitness = self.fitness_function(route)
            whale = Whale(route, fitness)
            self.population.append(whale)

            # Update best solution if necessary
            if fitness < self.best_fitness:
                self.best_fitness = fitness
                self.best_solution = whale.copy()

    def encircle_prey(self, whale, best_whale, a):
        """Perform encircling prey behavior (adapted for TSP)"""
        new_route = whale.route.copy()
        r = random()
        A = 2 * a * r - a  # coefficient vector A
        C = 2 * r  # coefficient vector C

        if abs(A) < 1:  # exploitation phase
            # Perform swap operations based on best solution
            num_swaps = max(1, int(abs(C) * self.city_count * 0.1))
            for _ in range(num_swaps):
                i, j = randint(0, self.city_count - 1), randint(0, self.city_count - 1)
                if best_whale.route[i] != new_route[i]:
                    idx = np.where(new_route == best_whale.route[i])[0][0]
                    new_route[i], new_route[idx] = new_route[idx], new_route[i]

        return new_route

    def bubble_net_attack(self, whale, best_whale, l):
        """Perform bubble-net attacking (exploitation phase)"""
        new_route = whale.route.copy()

        # Spiral updating position
        b = self.b  # spiral shape constant
        l = (random() * 2) - 1  # random number in [-1,1]

        # Calculate spiral path
        num_changes = int(abs(math.exp(b * l) * math.cos(2 * math.pi * l)) * self.city_count * 0.1)

        # Apply changes using 2-opt local search
        for _ in range(num_changes):
            i, j = sorted(np.random.choice(self.city_count, 2, replace=False))
            new_route[i:j + 1] = new_route[i:j + 1][::-1]

        return new_route

    def search_for_prey(self, whale):
        """Perform random search (exploration phase)"""
        new_route = whale.route.copy()

        # Random position update using insertion mutation
        num_insertions = randint(1, max(2, int(self.city_count * 0.1)))
        for _ in range(num_insertions):
            i, j = randint(0, self.city_count - 1), randint(0, self.city_count - 1)
            if i != j:
                value = new_route[i]
                new_route = np.delete(new_route, i)
                new_route = np.insert(new_route, j, value)

        return new_route

    def run(self):
        """Execute the Whale Optimization Algorithm"""
        # Initialize population
        self.initialize_population()

        # Main loop
        for iteration in range(self.max_iterations):
            # Update a linearly from 2 to 0
            a = 2 * (1 - iteration / self.max_iterations)

            # Update each whale's position
            for whale in self.population:
                # Generate random number for mechanism selection
                p = random()

                if p < 0.5:
                    # Encircling prey or searching for prey
                    if abs(a) < 1:
                        new_route = self.encircle_prey(whale, self.best_solution, a)
                    else:
                        new_route = self.search_for_prey(whale)
                else:
                    # Bubble-net attacking
                    new_route = self.bubble_net_attack(whale, self.best_solution, p)

                # Calculate fitness of new position
                new_fitness = self.fitness_function(new_route)

                # Update whale's position if better
                if new_fitness < whale.fitness:
                    whale.route = new_route
                    whale.fitness = new_fitness

                    # Update best solution if necessary
                    if new_fitness < self.best_fitness:
                        self.best_fitness = new_fitness
                        self.best_solution = Whale(new_route.copy(), new_fitness)

        return [self.best_solution.route], self.best_fitness