import os
import time
from multiprocessing import Pool
from typing import List, Tuple

import numpy as np

from common import utils
from common.interfaces import SearchProblemSolver, IteratedSearchProblemSolver
from solvers.local_search_improved import CandidateSteepSearch
from solvers.problem_solvers import GreedyCycleProblemSolver


def _get_unused_nodes(distance_matrix: np.ndarray, cycle: List[int]) -> List[int]:
    return [i for i in range(distance_matrix.shape[0]) if i not in cycle]


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

        process_pool.close()
        return best_cycle

    @staticmethod
    def _solve_single(distance_matrix: np.ndarray, problem_solver: SearchProblemSolver):
        cycle = problem_solver.solve(distance_matrix)
        cost = utils.calculate_path_length(distance_matrix, cycle)
        return cycle, cost


class IteratedLocalSearch1(IteratedSearchProblemSolver):
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


class IteratedLocalSearch2(IteratedSearchProblemSolver):
    def solve(self, distance_matrix: np.ndarray, max_time: float = 10.0, start_cycle=None) -> Tuple[List[int], int]:
        problem_solver = CandidateSteepSearch()
        best_cycle: List[int] = problem_solver.solve(distance_matrix) if not start_cycle else start_cycle
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
    def __perturb(distance_matrix: np.ndarray, cycle: List[int]) -> List[int]:
        cycle_ = cycle.copy()
        for _ in range(round(len(cycle_) * 0.20)):
            cycle_indices = list(range(len(cycle_) - 1))
            cycle_index = np.random.choice(cycle_indices)
            cycle_node = cycle_[cycle_index]
            cycle_.remove(cycle_node)
            if cycle_index == 0:
                cycle_[-1] = cycle_[0]

        cycle_ = GreedyCycleProblemSolver().solve(distance_matrix, start_cycle=cycle_)

        return cycle_


class IteratedLocalSearch2a(IteratedSearchProblemSolver):
    def solve(self, distance_matrix: np.ndarray, max_time: float = 10.0, start_cycle=None) -> Tuple[List[int], int]:
        problem_solver = GreedyCycleProblemSolver()
        best_cycle: List[int] = problem_solver.solve(distance_matrix)
        best_cost: int = utils.calculate_path_length(distance_matrix, best_cycle)

        local_search_invocation_count = 1
        time_start: float = time.time()
        duration = 0.0
        while duration < max_time:
            temp_cycle = self.__perturb(best_cycle)
            temp_cycle = problem_solver.solve(distance_matrix, start_cycle=temp_cycle)
            local_search_invocation_count += 1

            cost = utils.calculate_path_length(distance_matrix, temp_cycle)
            if cost < best_cost:
                best_cost = cost
                best_cycle = temp_cycle

            duration = time.time() - time_start

        return best_cycle, local_search_invocation_count

    @staticmethod
    def __perturb(cycle: List[int]) -> List[int]:
        cycle_ = cycle.copy()
        for _ in range(round(len(cycle_) * 0.20)):
            cycle_indices = list(range(len(cycle_) - 1))
            cycle_index = np.random.choice(cycle_indices)
            cycle_node = cycle_[cycle_index]
            cycle_.remove(cycle_node)
            if cycle_index == 0:
                while cycle_.count(cycle_node) > 0:
                    cycle_.remove(cycle_node)
                cycle_.append(cycle_[0])

        return cycle_
