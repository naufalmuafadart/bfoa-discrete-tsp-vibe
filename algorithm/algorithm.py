import numpy as np


class Algorithm:
    def __init__(self, dataset_name):
        """
        Initialize the Algorithm class with a dataset name.

        Args:
            dataset_name (str): Name of the dataset (without .npy extension)
        """
        self.dataset_name = dataset_name
        self.distance_matrix = np.load(f'./dataset/distance_matrix/{dataset_name}.npy')
        self.city_count = self.distance_matrix.shape[0]  # Get number of cities from matrix dimensions

    def fitness_function(self, route):
        """
        Calculate the total distance of a TSP route.

        Args:
            route (list or numpy.array): Permutation of cities (0 to city_count-1)

        Returns:
            float: Total distance of the route

        Raises:
            ValueError: If route is invalid (wrong length or contains invalid cities)
        """
        # Input validation
        if not isinstance(route, (list, np.ndarray)):
            raise ValueError("Route must be a list or numpy array")

        if len(route) != self.city_count:
            raise ValueError(f"Route must contain exactly {self.city_count} cities")

        if set(route) != set(range(self.city_count)):
            raise ValueError("Route must be a permutation of cities from 0 to city_count-1")

        # Calculate total distance
        total_distance = 0

        # Add distances between consecutive cities
        for i in range(self.city_count - 1):
            total_distance += self.distance_matrix[route[i]][route[i + 1]]

        # Add distance from last city back to first city to complete the circuit
        total_distance += self.distance_matrix[route[-1]][route[0]]

        return total_distance
