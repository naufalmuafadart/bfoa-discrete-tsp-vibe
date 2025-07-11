from collections import deque
import numpy as np
from random import random, randint
import math
from algorithm.algorithm import Algorithm


class TS(Algorithm):
    """Tabu Search for solving TSP"""

    def __init__(self, dataset_name, max_iterations=1, tabu_tenure=20):
        """
        Initialize the Tabu Search algorithm.

        Args:
            dataset_name (str): Name of the dataset.
            max_iterations (int): Maximum number of iterations for the search.
            tabu_tenure (int): The number of iterations a move is forbidden.
        """
        super().__init__(dataset_name)
        self.max_iterations = max_iterations
        self.tabu_tenure = tabu_tenure
        self.tabu_list = deque(maxlen=self.tabu_tenure)
        self.best_solution = None
        self.best_fitness = float('inf')

    def run(self):
        """Execute the Tabu Search algorithm."""
        # Start with a random initial solution
        current_route = np.random.permutation(self.city_count)
        self.best_solution = current_route
        self.best_fitness = self.fitness_function(current_route)

        self.tabu_list.clear()

        for _ in range(self.max_iterations):
            best_neighbor_route = None
            best_neighbor_fitness = float('inf')
            best_move = None

            # Explore the neighborhood of the current solution (using 2-opt swaps)
            for i in range(self.city_count):
                for j in range(i + 1, self.city_count):
                    # Define the move by the swapped indices
                    move = tuple(sorted((i, j)))

                    # Create the neighbor by reversing the segment
                    neighbor_route = current_route.copy()
                    neighbor_route[i:j + 1] = neighbor_route[i:j + 1][::-1]
                    neighbor_fitness = self.fitness_function(neighbor_route)

                    # Aspiration Criterion: Accept the move if it leads to a new best solution
                    aspiration_met = neighbor_fitness < self.best_fitness

                    # Accept the move if it's not in the tabu list or if it meets the aspiration criterion
                    if move not in self.tabu_list or aspiration_met:
                        if neighbor_fitness < best_neighbor_fitness:
                            best_neighbor_route = neighbor_route
                            best_neighbor_fitness = neighbor_fitness
                            best_move = move

            if best_neighbor_route is None:
                # If no improving move is found, the search might be stuck.
                # For this implementation, we continue, but diversification could be added here.
                continue

            # Move to the best neighbor found
            current_route = best_neighbor_route

            # Update the tabu list with the move that was just made
            if best_move:
                self.tabu_list.append(best_move)

            # Update the overall best solution if the new solution is better
            if best_neighbor_fitness < self.best_fitness:
                self.best_solution = best_neighbor_route
                self.best_fitness = best_neighbor_fitness

        return [self.best_solution], self.best_fitness
