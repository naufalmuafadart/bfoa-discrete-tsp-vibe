import numpy as np
from random import random, randint
from algorithm.algorithm import Algorithm

class Idea:
    """Represents an idea (a solution) in the Brain Storm Optimization algorithm."""

    def __init__(self, route, fitness):
        """
        Initialize an idea.
        Args:
            route (list): A permutation representing the TSP route.
            fitness (float): The fitness value of the route (total distance).
        """
        self.route = route
        self.fitness = fitness

    def copy(self):
        """Create a deep copy of the idea."""
        return Idea(self.route.copy(), self.fitness)


class BSO(Algorithm):
    """Brain Storm Optimization for solving TSP"""

    def __init__(self, dataset_name, population_size=20, max_iterations=10, num_clusters=5):
        """
        Initialize the BSO algorithm.

        Args:
            dataset_name (str): Name of the dataset.
            population_size (int): Size of the idea population.
            max_iterations (int): Maximum number of iterations.
            num_clusters (int): Number of clusters to group ideas into.
        """
        super().__init__(dataset_name)
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.num_clusters = num_clusters

        if self.population_size < self.num_clusters:
            raise ValueError("Population size must be greater than or equal to the number of clusters.")

        self.population = []
        self.global_best_solution = None

    def initialize_population(self):
        """Initialize a random population of ideas."""
        self.population = []
        for _ in range(self.population_size):
            route = np.random.permutation(self.city_count)
            fitness = self.fitness_function(route)
            self.population.append(Idea(route, fitness))

        self.population.sort(key=lambda x: x.fitness)
        self.global_best_solution = self.population[0].copy()

    def _cluster_ideas(self):
        """Group ideas into clusters and find their centers."""
        self.population.sort(key=lambda x: x.fitness)

        clusters = [[] for _ in range(self.num_clusters)]
        for i, idea in enumerate(self.population):
            clusters[i % self.num_clusters].append(idea)

        cluster_centers = [cluster[0] for cluster in clusters if cluster]
        return clusters, cluster_centers

    def _mutate(self, route):
        """Perform 2-opt mutation on a route."""
        mutated_route = route.copy()
        i, j = sorted(np.random.choice(len(mutated_route), 2, replace=False))
        mutated_route[i:j + 1] = mutated_route[i:j + 1][::-1]
        return mutated_route

    def _crossover(self, p1_route, p2_route):
        """Perform order crossover (OX1) between two parent routes."""
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

    def run(self):
        """Execute the Brain Storm Optimization algorithm."""
        self.initialize_population()

        p_one_cluster = 0.8  # Probability of generating a new idea from one cluster
        p_one_center = 0.5  # Probability of selecting the center from a cluster

        for _ in range(self.max_iterations):
            clusters, centers = self._cluster_ideas()

            new_population = []
            for i in range(self.population_size):
                if random() < p_one_cluster:
                    # Generate new idea from one cluster
                    cluster_idx = randint(0, len(centers) - 1)
                    selected_cluster = clusters[cluster_idx]

                    if random() < p_one_center:
                        base_idea = centers[cluster_idx]
                    else:
                        base_idea = selected_cluster[randint(0, len(selected_cluster) - 1)]

                    new_route = self._mutate(base_idea.route)
                else:
                    # Generate new idea from two clusters
                    idx1, idx2 = np.random.choice(len(centers), 2, replace=False)
                    idea1 = centers[idx1]
                    idea2 = centers[idx2]
                    new_route = self._crossover(idea1.route, idea2.route)

                new_fitness = self.fitness_function(new_route)
                new_idea = Idea(new_route, new_fitness)

                # Replace the original idea at this position with the new one
                if new_fitness < self.population[i].fitness:
                    new_population.append(new_idea)
                    if new_fitness < self.global_best_solution.fitness:
                        self.global_best_solution = new_idea.copy()
                else:
                    new_population.append(self.population[i])

            self.population = new_population

        return [self.global_best_solution.route], self.global_best_solution.fitness