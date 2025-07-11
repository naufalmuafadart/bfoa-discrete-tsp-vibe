import numpy as np
import random
import copy
from .algorithm import Algorithm


class Komodo:
    """A helper class to represent a single solution (a 'komodo') in the population."""
    def __init__(self, route, fitness):
        self.route = route
        self.fitness = fitness

    def copy(self):
        """Creates a deep copy of the komodo solution."""
        return Komodo(self.route.copy(), self.fitness)


class DKA(Algorithm):
    """
    Discrete Komodo Algorithm for solving the Traveling Salesperson Problem (TSP).

    This class implements the DKA metaheuristic with an interface consistent with
    the WOA class for comparison and benchmarking purposes.
    """

    def __init__(self, dataset_name, population_size=10, max_iterations=25,
                 big_male_ratio=0.3, female_ratio=0.4, destruction_ratio=0.25):
        """
        Initializes the DKA solver.

        Args:
            dataset_name (str): The name of the TSP dataset to solve.
            population_size (int): The number of solutions in the population.
            max_iterations (int): The total number of iterations to run.
            big_male_ratio (float): The proportion of the population designated as 'big males'.
            female_ratio (float): The proportion of the population designated as 'females'.
            destruction_ratio (float): The proportion of a route to remove during the 'ruin' phase.
        """
        super().__init__(dataset_name)
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.destruction_ratio = destruction_ratio

        # Calculate population distribution based on ratios
        self.n_big_male = int(population_size * big_male_ratio)
        self.n_female = int(population_size * female_ratio)
        self.n_small_male = population_size - self.n_big_male - self.n_female

        if self.n_big_male == 0 or self.n_small_male <= 0:
            raise ValueError("Invalid population ratios. Ensure there is at least one big male and one small male.")

        # Algorithm state variables
        self.population = []
        self.best_solution = None
        self.best_fitness = float('inf')

    def _initialize_population(self):
        """Creates the initial population of random solutions."""
        self.population = []
        for _ in range(self.population_size):
            # A solution is a random permutation of cities
            route = np.random.permutation(self.city_count)
            fitness = self.fitness_function(route)
            komodo = Komodo(route, fitness)
            self.population.append(komodo)

            # Keep track of the best solution found so far
            if fitness < self.best_fitness:
                self.best_fitness = fitness
                self.best_solution = komodo.copy()

    def _ruin_and_recreate(self, komodo_to_improve):
        """
        Applies the Ruin and Recreate heuristic to a given solution.
        1. Ruin: A portion of the route is destroyed by removing cities.
        2. Recreate: The removed cities are re-inserted using a greedy strategy.
        """
        # --- Ruin Phase ---
        partial_tour = komodo_to_improve.route.tolist()
        num_to_remove = int(self.city_count * self.destruction_ratio)
        if num_to_remove == 0:
            num_to_remove = 1

        # Randomly select and remove cities from the tour
        removed_cities = []
        for _ in range(num_to_remove):
            if not partial_tour:
                break
            city_to_remove = random.choice(partial_tour)
            partial_tour.remove(city_to_remove)
            removed_cities.append(city_to_remove)

        # --- Recreate Phase ---
        # Greedily insert the removed cities back into the partial tour
        for city in removed_cities:
            best_pos = -1
            min_increase = float('inf')

            # Find the insertion position that results in the minimum increase in tour length
            for i in range(len(partial_tour) + 1):
                # Calculate the cost increase of inserting 'city' at position 'i'
                prev_node = partial_tour[i - 1]
                next_node = partial_tour[i] if i < len(partial_tour) else partial_tour[0]

                cost_increase = (self.distance_matrix[prev_node][city] +
                                 self.distance_matrix[city][next_node] -
                                 self.distance_matrix[prev_node][next_node])

                if cost_increase < min_increase:
                    min_increase = cost_increase
                    best_pos = i

            partial_tour.insert(best_pos, city)

        return np.array(partial_tour)

    def run(self):
        """
        Executes the main loop of the Discrete Komodo Algorithm.

        Returns:
            tuple: A tuple containing the best route found and its corresponding fitness (distance).
        """
        self._initialize_population()

        for _ in range(self.max_iterations):
            # 1. Cluster Komodos: Sort by fitness and divide into social groups
            self.population.sort(key=lambda k: k.fitness)
            big_males = self.population[:self.n_big_male]
            small_males = self.population[self.n_big_male + self.n_female:]

            # 2. Select a 'big male' solution to improve
            komodo_to_improve = random.choice(big_males)

            # 3. Apply Ruin and Recreate to generate a new candidate solution
            new_route = self._ruin_and_recreate(komodo_to_improve)
            new_fitness = self.fitness_function(new_route)

            # 4. Replace the worst 'small male' if the new solution is better
            if new_fitness < small_males[-1].fitness:
                small_males[-1].route = new_route
                small_males[-1].fitness = new_fitness

                # Update the overall best solution if the new one is a global best
                if new_fitness < self.best_fitness:
                    self.best_fitness = new_fitness
                    self.best_solution.route = new_route
                    self.best_solution.fitness = new_fitness

        return self.best_solution.route, self.best_fitness
