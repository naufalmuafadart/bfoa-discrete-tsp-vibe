from random import random, sample
from algorithm.algorithm import Algorithm
import numpy as np

class GA(Algorithm):
    """Genetic Algorithm for solving TSP"""

    def __init__(self, dataset_name, population_size=10, max_iterations=5, crossover_rate=0.8, mutation_rate=0.2,
                 tournament_size=5):
        """
        Initialize the GA algorithm.

        Args:
            dataset_name (str): Name of the dataset
            population_size (int): Size of the population
            max_iterations (int): Maximum number of iterations
            crossover_rate (float): Probability of crossover
            mutation_rate (float): Probability of mutation
            tournament_size (int): Size of the tournament for selection
        """
        super().__init__(dataset_name)
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size

        self.population = []
        self.best_solution = None
        self.best_fitness = float('inf')

    def initialize_population(self):
        """Initialize random population"""
        self.population = []
        for _ in range(self.population_size):
            route = np.random.permutation(self.city_count)
            fitness = self.fitness_function(route)
            self.population.append({'route': route, 'fitness': fitness})

            if fitness < self.best_fitness:
                self.best_fitness = fitness
                self.best_solution = route.copy()

    def selection(self):
        """Selects parents using tournament selection."""
        parents = []
        for _ in range(self.population_size):
            tournament = sample(self.population, self.tournament_size)
            winner = min(tournament, key=lambda x: x['fitness'])
            parents.append(winner)
        return parents

    @staticmethod
    def partially_mapped_crossover(parent1, parent2):
        """
        Performs Partially Mapped Crossover (PMX) on two parents
        to create two offspring. This crossover operator is suitable for
        permutation-based chromosomes in Genetic Algorithms.

        Args:
            parent1 (np.ndarray): The first parent, a permutation array.
            parent2 (np.ndarray): The second parent, a permutation array.

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing the two generated offspring.
        """
        # Ensure parents are of the same length
        assert len(parent1) == len(parent2), "Parents must have the same length."
        size = len(parent1)

        # Initialize offspring with placeholders (-1 indicates an empty slot)
        offspring1 = np.full(size, -1, dtype=int)
        offspring2 = np.full(size, -1, dtype=int)

        # --- Step 1: Select a random crossover segment ---
        # Choose two random crossover points
        cx_point1 = np.random.randint(0, size)
        cx_point2 = np.random.randint(0, size)

        # Ensure the points are in order
        if cx_point1 > cx_point2:
            cx_point1, cx_point2 = cx_point2, cx_point1

        # Add one to the second point to make the slice inclusive
        cx_point2 += 1

        # --- Step 2: Copy the crossover segment from parents to offspring ---
        # Copy the segment from parent1 to offspring1
        offspring1[cx_point1:cx_point2] = parent1[cx_point1:cx_point2]
        # Copy the segment from parent2 to offspring2
        offspring2[cx_point1:cx_point2] = parent2[cx_point1:cx_point2]

        # --- Step 3: Create mappings from the crossover segments ---
        # This dictionary will hold the mapping for resolving duplicates.
        # For each element in parent1's segment, the corresponding element
        # in parent2's segment is its mapping, and vice-versa.
        mapping1 = {parent1[i]: parent2[i] for i in range(cx_point1, cx_point2)}
        mapping2 = {parent2[i]: parent1[i] for i in range(cx_point1, cx_point2)}

        # --- Step 4: Fill the remaining slots in the offspring ---
        for i in list(range(cx_point1)) + list(range(cx_point2, size)):
            # For Offspring 1
            # Get the element from parent2
            candidate = parent2[i]
            # If the candidate is already in the offspring1 (from the crossover segment),
            # follow the mapping chain until a valid element is found.
            while candidate in offspring1:
                candidate = mapping1[candidate]
            offspring1[i] = candidate

            # For Offspring 2
            # Get the element from parent1
            candidate = parent1[i]
            # If the candidate is already in the offspring2, follow the mapping chain.
            while candidate in offspring2:
                candidate = mapping2[candidate]
            offspring2[i] = candidate

        return offspring1, offspring2

    def mutation(self, route):
        """Performs swap mutation."""
        if random() < self.mutation_rate:
            i, j = sample(range(len(route)), 2)
            route[i], route[j] = route[j], route[i]
        return route

    def run(self):
        """Execute the Genetic Algorithm"""
        self.initialize_population()
        current_best = min(self.population, key=lambda x: x['fitness'])
        if current_best['fitness'] < self.best_fitness:
            self.best_fitness = current_best['fitness']
            self.best_solution = current_best['route'].copy()

        for _ in range(self.max_iterations):
            current_best = min(self.population, key=lambda x: x['fitness'])
            if current_best['fitness'] < self.best_fitness:
                self.best_fitness = current_best['fitness']
                self.best_solution = current_best['route'].copy()

            parents = self.selection()

            next_generation = [{'route': self.best_solution.copy(), 'fitness': self.best_fitness}]

            while len(next_generation) < self.population_size:
                p1, p2 = sample(parents, 2)

                if random() < self.crossover_rate:
                    c1_route, c2_route = self.partially_mapped_crossover(p1['route'], p2['route'])
                else:
                    c1_route, c2_route = p1['route'].copy(), p2['route'].copy()

                c1_route = self.mutation(c1_route)
                c2_route = self.mutation(c2_route)

                next_generation.append({'route': c1_route, 'fitness': self.fitness_function(c1_route)})
                if len(next_generation) < self.population_size:
                    next_generation.append({'route': c2_route, 'fitness': self.fitness_function(c2_route)})

            self.population = next_generation

        return [self.best_solution], self.best_fitness
