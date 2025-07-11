import numpy as np
from random import randint
from algorithm.algorithm import Algorithm


class Particle:
    """Represents a particle in the PSO algorithm"""

    def __init__(self, route, fitness):
        """
        Initialize a particle.

        Args:
            route (list): A permutation representing the TSP route
            fitness (float): The fitness value of the route (total distance)
        """
        self.route = route
        self.fitness = fitness
        self.best_route = route.copy()  # Personal best position
        self.best_fitness = fitness  # Personal best fitness

    def copy(self):
        """Create a deep copy of the particle"""
        particle = Particle(self.route.copy(), self.fitness)
        particle.best_route = self.best_route.copy()
        particle.best_fitness = self.best_fitness
        return particle


class PSO(Algorithm):
    """Particle Swarm Optimization algorithm for solving TSP"""

    def __init__(self, dataset_name, swarm_size=10, max_iterations=10, w=0.7, c1=2.0, c2=2.0):
        """
        Initialize the PSO algorithm.

        Args:
            dataset_name (str): Name of the dataset
            swarm_size (int): Number of particles in the swarm
            max_iterations (int): Maximum number of iterations
            w (float): Inertia weight
            c1 (float): Cognitive coefficient (personal best)
            c2 (float): Social coefficient (global best)
        """
        super().__init__(dataset_name)
        self.swarm_size = swarm_size
        self.max_iterations = max_iterations
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.best_solution = None
        self.best_fitness = float('inf')

    def initialize_swarm(self):
        """Initialize the swarm with random solutions"""
        particles = []
        for _ in range(self.swarm_size):
            route = np.random.permutation(self.city_count)
            fitness = self.fitness_function(route)
            particle = Particle(route, fitness)
            particles.append(particle)

            # Update global best if necessary
            if fitness < self.best_fitness:
                self.best_fitness = fitness
                self.best_solution = particle.copy()

        return particles

    def update_particle(self, particle):
        """Update particle's position using PSO movement rules adapted for TSP"""
        # Generate new position influenced by personal and global best
        new_route = particle.route.copy()

        # Apply swap operations based on personal and global best positions
        # Personal influence (c1)
        if np.random.random() < self.c1:
            # Perform swap operation based on personal best
            pos1, pos2 = randint(0, self.city_count - 1), randint(0, self.city_count - 1)
            new_route[pos1], new_route[pos2] = new_route[pos2], new_route[pos1]

        # Global influence (c2)
        if np.random.random() < self.c2:
            # Perform swap operation based on global best
            pos1, pos2 = randint(0, self.city_count - 1), randint(0, self.city_count - 1)
            new_route[pos1], new_route[pos2] = new_route[pos2], new_route[pos1]

        # Calculate fitness of new position
        new_fitness = self.fitness_function(new_route)

        # Update particle's position and personal best if improved
        if new_fitness < particle.fitness:
            particle.route = new_route
            particle.fitness = new_fitness
            if new_fitness < particle.best_fitness:
                particle.best_route = new_route.copy()
                particle.best_fitness = new_fitness

        return particle

    def run(self):
        """Execute the PSO algorithm"""
        # Initialize swarm
        particles = self.initialize_swarm()

        # Main loop
        for iteration in range(self.max_iterations):
            # Update each particle
            for particle in particles:
                updated_particle = self.update_particle(particle)

                # Update global best if necessary
                if updated_particle.fitness < self.best_fitness:
                    self.best_fitness = updated_particle.fitness
                    self.best_solution = updated_particle.copy()

        return self.best_solution.route, self.best_fitness
