import os
import random
import time
from multiprocessing import Pool
from typing import List, Tuple

import numpy as np

from common import utils
from common.interfaces import SearchProblemSolver, IteratedSearchProblemSolver
from solvers.local_search_improved import CandidateSteepSearch


def _get_unused_nodes(distance_matrix: np.ndarray, cycle: List[int]) -> List[int]:
    return [i for i in range(distance_matrix.shape[0]) if i not in cycle]


class HybridLocalSearch(IteratedSearchProblemSolver):
    # TODO: Przerobienie tego, bo to na razie skopiowane z poprzedniego zadania, poza __recombine()
    def solve(self, distance_matrix: np.ndarray, max_time: float = 10.0, start_cycle=None) -> Tuple[List[int], int]:
        problem_solver = CandidateSteepSearch()
        best_cycle: List[int] = problem_solver.solve(distance_matrix)
        best_cost: int = utils.calculate_path_length(distance_matrix, best_cycle)

        local_search_invocation_count = 1
        time_start: float = time.time()
        duration = 0.0

        while duration < max_time:
            temp_cycle = self.__perturb(distance_matrix, best_cycle)
            temp_cycle = problem_solver.solve(distance_matrix, temp_cycle)
            local_search_invocation_count += 1

            cost = utils.calculate_path_length(distance_matrix, temp_cycle)
            if cost < best_cost:
                best_cost = cost
                best_cycle = temp_cycle

            duration = time.time() - time_start

        return best_cycle, local_search_invocation_count

    @staticmethod
    def __recombine(cycle1: List[int], cycle2: List[int]) -> List[int]:
        cycle1_ = cycle1.copy()
        cycle2_ = cycle2.copy()

        child_cycle = [x for x in cycle1_ if x in cycle2_]
        rest_nodes = [x for x in cycle1_ if x not in child_cycle] + [x for x in cycle2_ if x not in child_cycle]
        while len(child_cycle) < len(cycle1_):
            new_node = random.choice(rest_nodes)
            child_cycle.append(new_node)
            rest_nodes.remove(new_node)
        return child_cycle

    @staticmethod
    # TODO: To nie wiem czy się przyda w ogóle, ale skopiowałem na wszelki
    def __perturb(distance_matrix: np.ndarray, cycle: List[int]) -> List[int]:
        cycle_ = cycle.copy()
        unused_nodes = _get_unused_nodes(distance_matrix, cycle_)

        unused_node_indices = list(range(len(unused_nodes)))
        cycle_indices = list(range(len(cycle_) - 1))

        for _ in range(5):
            unused_index = np.random.choice(unused_node_indices)
            cycle_index = np.random.choice(cycle_indices)
            cycle_[cycle_index], unused_nodes[unused_index] = unused_nodes[unused_index], cycle_[cycle_index]

            if cycle_index == 0:
                cycle_[-1] = cycle_[0]

        return cycle_