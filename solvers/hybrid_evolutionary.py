import time
from typing import List, Tuple

import numpy as np

from common import utils
from common.interfaces import IteratedSearchProblemSolver
from solvers.local_search import RandomSearch
from solvers.local_search_improved import CandidateSteepSearch


def _get_unused_nodes(distance_matrix: np.ndarray, cycle: List[int]) -> List[int]:
    return [i for i in range(distance_matrix.shape[0]) if i not in cycle]


class HybridEvolutionarySolver(IteratedSearchProblemSolver):
    POPULATION_SIZE = 20

    def solve(self, distance_matrix: np.ndarray, max_time: float = 10.0, start_cycle=None) -> Tuple[List[int], int]:
        problem_solver = CandidateSteepSearch()
        random_problem_solver = RandomSearch()

        # Generate random population
        population: List[List[int]] = []
        while len(population) < HybridEvolutionarySolver.POPULATION_SIZE:
            solution = random_problem_solver.solve(distance_matrix)
            if solution not in population:
                population.append(solution)

        population_costs = [utils.calculate_path_length(distance_matrix, solution) for solution in population]

        local_search_invocation_count = 1
        time_start: float = time.time()
        duration = 0.0
        while duration < max_time:
            # Pick random parents
            parent_1 = population[np.random.randint(HybridEvolutionarySolver.POPULATION_SIZE)]
            parent_2 = parent_1
            while parent_2 == parent_1:
                parent_2 = population[np.random.randint(HybridEvolutionarySolver.POPULATION_SIZE)]

            # Recombine
            child = self.__recombine(parent_1, parent_2)

            # Improve child with local search
            child = problem_solver.solve(distance_matrix, child)
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

        best_index = np.argmax(population_costs)

        return population[best_index], local_search_invocation_count

    @staticmethod
    def __recombine(parent_1: List[int], parent_2: List[int]) -> List[int]:
        parent_1_ = parent_1.copy()
        parent_2_ = parent_2.copy()
        del parent_1_[-1]
        del parent_2_[-1]

        target_length = len(parent_1_)

        sub_paths = []
        sub_path = []
        for i in range(target_length):
            if parent_1_[i] == parent_2_[i]:
                sub_path.append(parent_1_[i])
            elif sub_path:
                sub_paths.append(sub_path)
                sub_path = []

        if sub_path:
            sub_paths.append(sub_path)

        child_cycle = []
        while len(sub_paths) != 0:
            random_index = np.random.randint(0, len(sub_paths))
            sub_path = sub_paths[random_index]
            child_cycle += sub_path
            del sub_paths[random_index]

        for i in range(target_length):
            if len(child_cycle) >= target_length:
                break

            parent_node = parent_1_[i]
            if parent_node not in child_cycle:
                child_cycle.append(parent_node)

        return child_cycle + [child_cycle[0]]
