import numpy as np
from random import randint
from algorithm.algorithm import Algorithm


class FoodSource:
    """Represents a food source in the Artificial Bee Colony algorithm"""

    def __init__(self, route, fitness):
        """
        Initialize a food source.

        Args:
            route (list): A permutation representing the TSP route
            fitness (float): The fitness value of the route (total distance)
        """
        self.route = route
        self.fitness = fitness
        self.trials = 0  # Counter for abandonment

    def copy(self):
        """Create a deep copy of the food source"""
        return FoodSource(self.route.copy(), self.fitness)


class ABC(Algorithm):
    """Artificial Bee Colony algorithm for solving TSP"""

    def __init__(self, dataset_name, colony_size=10, max_iterations=10, limit=20):
        """
        Initialize the ABC algorithm.

        Args:
            dataset_name (str): Name of the dataset
            colony_size (int): Number of food sources (half of the colony size)
            max_iterations (int): Maximum number of iterations
            limit (int): Limit of trials before abandoning a food source
        """
        super().__init__(dataset_name)
        self.colony_size = colony_size  # Number of food sources
        self.max_iterations = max_iterations
        self.limit = limit  # Abandonment limit
        self.best_solution = None
        self.best_fitness = float('inf')

    def initialize_population(self):
        """Initialize the food sources randomly"""
        food_sources = []
        for _ in range(self.colony_size):
            # Generate random permutation
            route = np.random.permutation(self.city_count)
            fitness = self.fitness_function(route)
            food_source = FoodSource(route, fitness)
            food_sources.append(food_source)

            # Update best solution if necessary
            if fitness < self.best_fitness:
                self.best_fitness = fitness
                self.best_solution = food_source.copy()

        return food_sources

    def employed_bee_phase(self, food_sources):
        """Employed bee phase of ABC algorithm"""
        for i in range(len(food_sources)):
            # Generate neighbor solution using swap operation
            new_route = food_sources[i].route.copy()
            # Select two random positions and swap them
            pos1, pos2 = randint(0, self.city_count - 1), randint(0, self.city_count - 1)
            new_route[pos1], new_route[pos2] = new_route[pos2], new_route[pos1]

            # Calculate fitness of new solution
            new_fitness = self.fitness_function(new_route)

            # Apply greedy selection
            if new_fitness < food_sources[i].fitness:
                food_sources[i] = FoodSource(new_route, new_fitness)
                if new_fitness < self.best_fitness:
                    self.best_fitness = new_fitness
                    self.best_solution = food_sources[i].copy()
            else:
                food_sources[i].trials += 1

    def onlooker_bee_phase(self, food_sources):
        """Onlooker bee phase of ABC algorithm"""
        # Calculate probabilities
        total_fitness = sum(1 / fs.fitness for fs in food_sources)
        probabilities = [(1 / fs.fitness) / total_fitness for fs in food_sources]

        for _ in range(self.colony_size):
            # Select food source based on probability
            selected_idx = np.random.choice(len(food_sources), p=probabilities)

            # Generate neighbor solution
            new_route = food_sources[selected_idx].route.copy()
            pos1, pos2 = randint(0, self.city_count - 1), randint(0, self.city_count - 1)
            new_route[pos1], new_route[pos2] = new_route[pos2], new_route[pos1]

            # Calculate fitness
            new_fitness = self.fitness_function(new_route)

            # Apply greedy selection
            if new_fitness < food_sources[selected_idx].fitness:
                food_sources[selected_idx] = FoodSource(new_route, new_fitness)
                if new_fitness < self.best_fitness:
                    self.best_fitness = new_fitness
                    self.best_solution = food_sources[selected_idx].copy()
            else:
                food_sources[selected_idx].trials += 1

    def scout_bee_phase(self, food_sources):
        """Scout bee phase of ABC algorithm"""
        for i in range(len(food_sources)):
            if food_sources[i].trials >= self.limit:
                # Generate new random solution
                new_route = np.random.permutation(self.city_count)
                new_fitness = self.fitness_function(new_route)
                food_sources[i] = FoodSource(new_route, new_fitness)

                if new_fitness < self.best_fitness:
                    self.best_fitness = new_fitness
                    self.best_solution = food_sources[i].copy()

    def run(self):
        """Execute the ABC algorithm"""
        # Initialize population
        food_sources = self.initialize_population()

        # Main loop
        for iteration in range(self.max_iterations):
            # Employed bee phase
            self.employed_bee_phase(food_sources)

            # Onlooker bee phase
            self.onlooker_bee_phase(food_sources)

            # Scout bee phase
            self.scout_bee_phase(food_sources)

        return self.best_solution.route, self.best_fitness
