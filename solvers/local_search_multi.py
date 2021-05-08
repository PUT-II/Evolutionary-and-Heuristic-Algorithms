import os
import time
from multiprocessing import Pool
from typing import List

import numpy as np

from common import utils
from common.interfaces import SearchProblemSolver
from solvers.local_search_improved import CandidateSteepSearch


class MultipleStartLocalSearch(SearchProblemSolver):
    def solve(self, distance_matrix: np.ndarray, start_cycle=None, max_time=None) -> List[int]:
        best_cost: int = np.iinfo(np.int32).max
        best_cycle: List[int] = []

        problem_solver = CandidateSteepSearch()
        process_pool = Pool(processes=os.cpu_count())
        process_res_list = []
        for _ in range(100):
            res = process_pool.apply_async(self._solve_single, (distance_matrix, problem_solver))
            process_res_list.append(res)

        for proc_res in process_res_list:
            cycle, cost = proc_res.get()

            if cost < best_cost:
                best_cost = cost
                best_cycle = cycle

        return best_cycle

    @staticmethod
    def _solve_single(distance_matrix: np.ndarray, problem_solver: SearchProblemSolver):
        cycle = problem_solver.solve(distance_matrix)
        cost = utils.calculate_path_length(distance_matrix, cycle)
        return cycle, cost


class IteratedLocalSearch1(SearchProblemSolver):
    def solve(self, distance_matrix: np.ndarray, max_time: float = 10.0, start_cycle=None) -> List[int]:
        problem_solver = CandidateSteepSearch()
        cycle = problem_solver.solve(distance_matrix)

        best_cost: int = utils.calculate_path_length(distance_matrix, cycle)
        best_cycle: List[int] = cycle

        time_start: float = time.time()
        duration = 0.0
        while duration < max_time:
            cycle = self.__perturb(distance_matrix, cycle)
            cycle = problem_solver.solve(distance_matrix, cycle)

            cost = utils.calculate_path_length(distance_matrix, cycle)
            if cost < best_cost:
                best_cost = cost
                best_cycle = cycle

            duration = time.time() - time_start

        return best_cycle

    @staticmethod
    def __get_unused_nodes(distance_matrix: np.ndarray, cycle: List[int]) -> List[int]:
        return [i for i in range(distance_matrix.shape[0]) if i in cycle]

    def __perturb(self, distance_matrix: np.ndarray, cycle: List[int]) -> List[int]:
        unused_nodes = self.__get_unused_nodes(distance_matrix, cycle)

        unused_node_indices = list(range(len(unused_nodes)))
        cycle_indices = list(range(len(cycle)))
        cycle_indices.remove(0)
        cycle_indices.remove(len(cycle) - 1)

        for _ in range(5):
            unused_index = np.random.choice(unused_node_indices)
            cycle_index = np.random.choice(cycle_indices)
            cycle[cycle_index], unused_nodes[unused_index] = unused_nodes[unused_index], cycle[cycle_index]

        return cycle
