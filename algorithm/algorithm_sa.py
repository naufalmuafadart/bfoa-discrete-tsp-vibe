import numpy as np
import math
from random import randint, random
from algorithm.algorithm import Algorithm


class Solution:
    """Represents a solution in the Simulated Annealing algorithm"""

    def __init__(self, route, fitness):
        """
        Initialize a solution.

        Args:
            route (list): A permutation representing the TSP route
            fitness (float): The fitness value of the route (total distance)
        """
        self.route = route
        self.fitness = fitness

    def copy(self):
        """Create a deep copy of the solution"""
        return Solution(self.route.copy(), self.fitness)


class SA(Algorithm):
    """Simulated Annealing algorithm for solving TSP"""

    def __init__(self, dataset_name, initial_temp=2, final_temp=1.0,
                 cooling_rate=0.95, iterations_per_temp=2):
        """
        Initialize the SA algorithm.

        Args:
            dataset_name (str): Name of the dataset
            initial_temp (float): Initial temperature
            final_temp (float): Final temperature (stopping condition)
            cooling_rate (float): Rate at which temperature decreases
            iterations_per_temp (int): Number of iterations at each temperature
        """
        super().__init__(dataset_name)
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.cooling_rate = cooling_rate
        self.iterations_per_temp = iterations_per_temp
        self.best_solution = None
        self.best_fitness = float('inf')
        self.current_solution = None

    def initialize_solution(self):
        """Initialize a random solution"""
        route = np.random.permutation(self.city_count)
        fitness = self.fitness_function(route)
        self.current_solution = Solution(route, fitness)
        self.best_solution = self.current_solution.copy()
        self.best_fitness = fitness

    def generate_neighbor(self):
        """Generate a neighbor solution using one of three neighborhood operators"""
        # Randomly choose a neighborhood operator
        operator = randint(1, 3)
        new_route = self.current_solution.route.copy()

        if operator == 1:
            # Swap operator: swap two random cities
            i, j = randint(0, self.city_count - 1), randint(0, self.city_count - 1)
            new_route[i], new_route[j] = new_route[j], new_route[i]

        elif operator == 2:
            # Reverse operator: reverse a subsequence
            i, j = sorted([randint(0, self.city_count - 1), randint(0, self.city_count - 1)])
            new_route[i:j + 1] = new_route[i:j + 1][::-1]

        else:
            # Insert operator: remove from one position and insert at another
            i, j = randint(0, self.city_count - 1), randint(0, self.city_count - 1)
            city = new_route[i]
            new_route = np.delete(new_route, i)
            new_route = np.insert(new_route, j, city)

        return Solution(new_route, self.fitness_function(new_route))

    def acceptance_probability(self, current_fitness, new_fitness, temperature):
        """Calculate probability of accepting a worse solution"""
        if new_fitness < current_fitness:
            return 1.0
        return math.exp((current_fitness - new_fitness) / temperature)

    def run(self):
        """Execute the Simulated Annealing algorithm"""
        # Initialize solution
        self.initialize_solution()

        # Set initial temperature
        temperature = self.initial_temp

        # Main loop
        while temperature > self.final_temp:
            # Inner loop for current temperature
            for _ in range(self.iterations_per_temp):
                # Generate neighbor solution
                neighbor = self.generate_neighbor()

                # Calculate acceptance probability
                if random() < self.acceptance_probability(
                        self.current_solution.fitness,
                        neighbor.fitness,
                        temperature
                ):
                    self.current_solution = neighbor

                    # Update best solution if necessary
                    if neighbor.fitness < self.best_fitness:
                        self.best_solution = neighbor.copy()
                        self.best_fitness = neighbor.fitness

            # Cool down
            temperature *= self.cooling_rate

        return self.best_solution.route, self.best_fitness
