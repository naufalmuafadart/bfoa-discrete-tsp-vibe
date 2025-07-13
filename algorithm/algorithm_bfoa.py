import random
from enum import Enum
from algorithm.algorithm import Algorithm
from algorithm.algorithm_ga import GA
import numpy as np


class Agent:
    """Represents a single agent in the BFOA, corresponding to one solution."""

    def __init__(self, agent_length, fitness_function, is_maximizing=True):
        self.agent_length = agent_length
        self.fitness_function = fitness_function
        self.is_maximizing = is_maximizing
        self.fitness_value = None
        self.vector = None

        self.generate_random_vector(agent_length)
        self.calculate_fitness()

    def generate_random_vector(self, city_count):
        """Creates a random initial solution (tour)."""
        self.vector = np.random.permutation(city_count)

    def calculate_fitness(self):
        """Calculates and updates the fitness of the agent's current vector."""
        self.fitness_value = self.fitness_function(self.vector)

    def airplane_movement(self):
        """Performs airplane movement using scramble mutation to explore the search space."""
        n = len(self.vector)
        i, j = sorted(random.sample(range(n), 2))
        subsequence = self.vector[i:j+1]
        random.shuffle(subsequence)
        self.vector[i:j+1] = subsequence
        self.calculate_fitness()

    def builder_movement(self):
        """Performs builder movement using insert mutation for local refinement."""
        # Make a copy to avoid modifying the original array
        mutated_arr = self.vector
        n = len(mutated_arr)

        # 1. Randomly select the index of the element (gene) to move.
        # np.random.randint is exclusive of the high value, so it's perfect for 0-based indexing.
        gene_index = np.random.randint(0, n)

        # 2. Get the value of the element to move.
        gene_to_move = mutated_arr[gene_index]

        # 3. Delete the selected element from the array.
        # np.delete returns a new array with the element removed.
        temp_arr = np.delete(mutated_arr, gene_index)

        # 4. Randomly select a new position to insert the element.
        # The new position can be anywhere from the start (0) to the end (n-1).
        # Since temp_arr has n-1 elements, the insertion index can be from 0 to n-1.
        insertion_point = np.random.randint(0, n)

        # 5. Insert the element into the new position.
        # np.insert returns the final mutated array.
        self.vector = np.insert(temp_arr, insertion_point, gene_to_move)
        self.calculate_fitness()

    def commander_movement(self, enemy_commander_vector):
        """Performs Partially Mapped Crossover (PMX) with an enemy commander's vector."""
        offspring1, _ = GA.partially_mapped_crossover(self.vector, enemy_commander_vector)
        self.vector = offspring1
        self.calculate_fitness()

    def cavalry_movement(self):
        """Performs flanking movement by rotating a segment of the tour."""
        n = len(self.vector)
        if n < 2:
            return

        i, j = sorted(random.sample(range(n), 2))
        segment = self.vector[i:j + 1]

        if len(segment) > 1:
            # A random rotation amount, ensuring it's not a full circle.
            rotation = random.randint(1, len(segment) - 1)

            # Use np.roll for efficient rotation of the segment.
            # A negative shift is used for a left rotation.
            self.vector[i:j + 1] = np.roll(segment, shift=-rotation)
            self.calculate_fitness()

    def special_force_movement(self):
        """Performs targeted insertion by finding the best position for a random city."""
        n = len(self.vector)
        if n <= 2:
            return  # Not enough cities to perform an insertion

        # Randomly select a city to move
        remove_idx = random.randint(0, n - 1)
        city_to_move = self.vector[remove_idx]
        temp_vector = np.delete(self.vector, remove_idx)

        best_fitness = float('inf')
        best_vector = None

        # Try inserting the city at every possible position to find the best one
        for i in range(n):
            # Create a candidate tour by inserting the city
            candidate_vector = np.insert(temp_vector, i, city_to_move)
            candidate_fitness = self.fitness_function(candidate_vector)

            # If this insertion is better, save it
            if candidate_fitness < best_fitness:
                best_fitness = candidate_fitness
                best_vector = candidate_vector

        # Update the agent's tour and fitness if a better one was found
        if best_vector is not None:
            self.vector = best_vector
            self.fitness_value = best_fitness


class SquadMode(Enum):
    ATTACKING = 1
    DEFENDING = 2


class Squad:
    """Represents a squad of agents with different roles."""

    def __init__(self, mode):
        self.mode = mode
        self.air_forces = []
        self.commander = None
        self.left_cavalry = None
        self.right_cavalry = None
        self.special_force = None
        self.builder = None

    def assign_squad(self, is_maximizing):
        """Sorts agents by fitness and assigns them to roles."""
        self.air_forces.sort(key=lambda agent: agent.fitness_value, reverse=is_maximizing)
        self.commander = self.air_forces[0]
        self.left_cavalry = self.air_forces[1]
        self.right_cavalry = self.air_forces[2]
        self.special_force = self.air_forces[3]
        self.builder = self.air_forces[4]


class BFOA(Algorithm):
    """
    Battlefield Optimization Algorithm for solving TSP.
    This class is designed with an interface compatible with the WOA class.
    """

    def __init__(self, dataset_name, population_size=10, max_iterations=2, b=1.0):
        super().__init__(dataset_name)
        if population_size < 10 or population_size % 2 != 0:
            raise ValueError("Population size must be an even number and at least 10 for BFOA.")

        self.max_iterations = max_iterations
        self.n_plane = population_size // 2  # Agents per squad
        self.is_maximizing = False  # For TSP, we minimize distance

        self.phase_1_max_iter = 2  # Initial exploration phase
        self.squad1 = Squad(SquadMode.ATTACKING)
        self.squad2 = Squad(SquadMode.DEFENDING)
        self.best_troops = None
        self.best_troops_fitness = float('inf')

    def _update_best_troops(self):
        """Finds the best agent between the two squad commanders and updates the global best."""
        s1_commander = self.squad1.commander
        s2_commander = self.squad2.commander

        current_best_commander = s1_commander if s1_commander.fitness_value < s2_commander.fitness_value else s2_commander

        if current_best_commander.fitness_value < self.best_troops_fitness:
            self.best_troops = current_best_commander.vector.copy()
            self.best_troops_fitness = current_best_commander.fitness_value

    def run(self):
        """Executes the Battlefield Optimization Algorithm."""
        # Phase 1: Initialization and Reconnaissance
        for _ in range(self.n_plane):
            self.squad1.air_forces.append(Agent(self.city_count, self.fitness_function, self.is_maximizing))
            self.squad2.air_forces.append(Agent(self.city_count, self.fitness_function, self.is_maximizing))

        for _ in range(self.phase_1_max_iter):
            for agent in self.squad1.air_forces + self.squad2.air_forces:
                agent.airplane_movement()

        self.squad1.assign_squad(self.is_maximizing)
        self.squad2.assign_squad(self.is_maximizing)
        self._update_best_troops()

        # Phase 2: Main Battle Loop
        for _ in range(self.max_iterations):
            # Determine attacking and defending squads for this iteration
            s1_is_better = self.squad1.commander.fitness_value < self.squad2.commander.fitness_value
            attacking_squad = self.squad2 if s1_is_better else self.squad1
            defending_squad = self.squad1 if s1_is_better else self.squad2

            # Defending squad refines its positions
            defending_squad.commander.builder_movement()
            defending_squad.left_cavalry.builder_movement()
            defending_squad.right_cavalry.builder_movement()
            defending_squad.builder.builder_movement()

            # Attacking squad engages the defending commander
            attacking_squad.commander.commander_movement(defending_squad.commander.vector)
            attacking_squad.left_cavalry.cavalry_movement()
            attacking_squad.right_cavalry.cavalry_movement()
            attacking_squad.builder.builder_movement()

            # Special forces from both squads perform their movements
            self.squad1.special_force.special_force_movement()
            self.squad2.special_force.special_force_movement()

            # Re-evaluate fitness and re-assign roles
            self.squad1.assign_squad(self.is_maximizing)
            self.squad2.assign_squad(self.is_maximizing)

            # Update the global best solution found so far
            self._update_best_troops()

        return [self.best_troops], self.best_troops_fitness
