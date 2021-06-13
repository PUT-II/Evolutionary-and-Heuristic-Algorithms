import time
from typing import List, Tuple

import numpy as np

import common.utils as utils
from common.interfaces import IteratedSearchProblemSolver
from solvers.local_search import RandomSearch
from solvers.local_search_multi import IteratedLocalSearch2


class ImprovedHybridEvolutionarySolver(IteratedSearchProblemSolver):
    POPULATION_SIZE = 4

    def solve(self, distance_matrix: np.ndarray, max_time: float = 10.0, start_cycle=None) -> Tuple[List[int], int]:
        problem_solver = IteratedLocalSearch2()
        random_problem_solver = RandomSearch()

        # Generate random population
        population: List[List[int]] = []
        while len(population) < self.POPULATION_SIZE:
            solution = random_problem_solver.solve(distance_matrix)

            if solution not in population:
                population.append(solution)

        population_costs = [utils.calculate_path_length(distance_matrix, solution) for solution in population]

        local_search_invocation_count = 0
        time_start: float = time.time()
        duration = 0.0
        while duration < max_time:
            duration = time.time() - time_start

            # Pick random parents
            parent_1 = population[np.random.randint(self.POPULATION_SIZE)]
            parent_2 = parent_1
            while parent_2 == parent_1:
                parent_2 = population[np.random.randint(self.POPULATION_SIZE)]

            # Recombine
            child = self.__recombine(parent_1, parent_2)

            # Improve child with local search
            # child = problem_solver.solve(distance_matrix, start_cycle=child)
            child = problem_solver.solve(distance_matrix, 2.5, child)[0]
            local_search_invocation_count += 1

            # Ignore if child already in population
            if child in population:
                continue

            # Search for worse solution than child
            child_cost = utils.calculate_path_length(distance_matrix, child)

            worst_cost = child_cost
            worst_index = -1
            for i, cost in enumerate(population_costs):
                if cost > worst_cost:
                    worst_index = i
                    worst_cost = cost

            # If no worse solution found then ignore child
            if worst_index == -1:
                continue

            # Replace worse solution with child
            population[worst_index] = child
            population_costs[worst_index] = child_cost
            duration = time.time() - time_start

        best_index = np.argmin(population_costs)

        return population[best_index], local_search_invocation_count

    @staticmethod
    def __recombine(parent_1: List[int], parent_2: List[int]) -> List[int]:
        parent_1_ = parent_1.copy()
        parent_2_ = parent_2.copy()

        # Remove last element of cycles (make paths)
        del parent_1_[-1]
        del parent_2_[-1]

        target_length = len(parent_1_)

        # Find common sub-paths in parent paths
        sub_paths = []
        sub_path = []

        zipped_parents = zip(parent_1_, parent_2_)
        if np.random.random() > 0.5:
            zipped_parents = reversed(list(zipped_parents))

        for parent_node_1, parent_node_2 in zipped_parents:
            if parent_node_1 == parent_node_2:
                sub_path.append(parent_node_1)
            elif sub_path:
                sub_paths.append(sub_path)
                sub_path = []

        if sub_path:
            sub_paths.append(sub_path)

        # Randomly construct child path from sub-paths
        child_path = []
        while len(sub_paths) != 0:
            random_index = np.random.randint(0, len(sub_paths))
            child_path += sub_paths[random_index]
            del sub_paths[random_index]

        # Fill missing nodes if nodes from parent 1
        for parent_node in parent_1_:
            if len(child_path) >= target_length:
                break

            if parent_node not in child_path:
                child_path.append(parent_node)

        # Create cycle from child path
        return child_path + [child_path[0]]
