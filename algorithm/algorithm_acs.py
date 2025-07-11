import numpy as np
from algorithm.algorithm import Algorithm

class ACS(Algorithm):
    """Ant Colony System for solving TSP"""

    def __init__(self, dataset_name, num_ants=10, max_iterations=5, alpha=1.0, beta=2.0, rho=0.1, q0=0.9):
        """
        Initialize the Ant Colony System algorithm.

        Args:
            dataset_name (str): Name of the dataset.
            num_ants (int): Number of ants in the colony.
            max_iterations (int): Maximum number of iterations.
            alpha (float): Pheromone influence factor.
            beta (float): Heuristic information (distance) influence factor.
            rho (float): Pheromone evaporation rate.
            q0 (float): Probability of choosing the best next city (exploitation).
        """
        super().__init__(dataset_name)
        self.num_ants = num_ants
        self.max_iterations = max_iterations
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.q0 = q0

        self.heuristic_info = 1 / (self.distance_matrix + np.finfo(np.float64).eps)
        self.pheromone_matrix = None
        self.best_solution = None
        self.best_fitness = float('inf')

    def _initialize_pheromones(self):
        """Initialize the pheromone matrix based on a nearest-neighbor tour."""
        _, nn_fitness = self._get_nearest_neighbor_tour()
        initial_pheromone = self.num_ants / nn_fitness
        self.pheromone_matrix = np.full((self.city_count, self.city_count), initial_pheromone)

    def _get_nearest_neighbor_tour(self):
        """Create a tour using the nearest neighbor heuristic."""
        start_city = np.random.randint(self.city_count)
        current_city = start_city
        unvisited = list(range(self.city_count))
        unvisited.remove(start_city)
        tour = [start_city]

        while unvisited:
            nearest_city = min(unvisited, key=lambda city: self.distance_matrix[current_city][city])
            tour.append(nearest_city)
            unvisited.remove(nearest_city)
            current_city = nearest_city

        return np.array(tour), self.fitness_function(np.array(tour))

    def run(self):
        """Execute the Ant Colony System algorithm."""
        self._initialize_pheromones()
        self.best_solution, self.best_fitness = self._get_nearest_neighbor_tour()

        for _ in range(self.max_iterations):
            for _ in range(self.num_ants):
                tour = self._construct_tour()
                fitness = self.fitness_function(tour)

                if fitness < self.best_fitness:
                    self.best_solution = tour
                    self.best_fitness = fitness

            self._global_pheromone_update()

        return [self.best_solution], self.best_fitness

    def _construct_tour(self):
        """Construct a tour for a single ant."""
        start_city = np.random.randint(self.city_count)
        tour = [start_city]
        unvisited = set(range(self.city_count))
        unvisited.remove(start_city)

        while unvisited:
            current_city = tour[-1]
            next_city = self._select_next_city(current_city, list(unvisited))
            tour.append(next_city)
            unvisited.remove(next_city)
            self._local_pheromone_update(current_city, next_city)

        return np.array(tour)

    def _select_next_city(self, current_city, unvisited_cities):
        """Select the next city for an ant using the ACS transition rule."""
        pheromone = self.pheromone_matrix[current_city, unvisited_cities]
        heuristic = self.heuristic_info[current_city, unvisited_cities]
        attraction = np.power(pheromone, self.alpha) * np.power(heuristic, self.beta)

        if np.random.rand() < self.q0:
            next_city_index = np.argmax(attraction)
        else:
            probabilities = attraction / np.sum(attraction)
            next_city_index = np.random.choice(len(unvisited_cities), p=probabilities)

        return unvisited_cities[next_city_index]

    def _local_pheromone_update(self, city1, city2):
        """Apply the local pheromone update."""
        initial_pheromone = 1.0 / (self.city_count * self.best_fitness) if self.best_fitness != float('inf') else 1e-5
        self.pheromone_matrix[city1, city2] = (1 - self.rho) * self.pheromone_matrix[
            city1, city2] + self.rho * initial_pheromone
        self.pheromone_matrix[city2, city1] = self.pheromone_matrix[city1, city2]

    def _global_pheromone_update(self):
        """Apply the global pheromone update to the best tour."""
        self.pheromone_matrix *= (1 - self.rho)
        deposit_amount = 1.0 / self.best_fitness

        for i in range(self.city_count):
            city1 = self.best_solution[i]
            city2 = self.best_solution[(i + 1) % self.city_count]
            self.pheromone_matrix[city1, city2] += self.rho * deposit_amount
            self.pheromone_matrix[city2, city1] += self.rho * deposit_amount
