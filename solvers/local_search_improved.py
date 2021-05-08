from enum import Enum
from typing import List

import numpy as np

from common.interfaces import SearchProblemSolver
from solvers.local_search import RandomSearch


def _find_best_outside_swap(distance_matrix: np.ndarray, cycle: list, unused_nodes_list: list, i_1):
    best_move: tuple = tuple()
    best_cost_delta = np.iinfo(distance_matrix.dtype).max

    for i_2 in range(len(unused_nodes_list)):
        move, cost_delta = _find_outside_swap_move(distance_matrix, cycle, unused_nodes_list, i_1, i_2)
        if cost_delta < best_cost_delta:
            best_cost_delta = cost_delta
            best_move = move
    return best_move, best_cost_delta


def _find_best_outside_swap_candidate(distance_matrix: np.ndarray, cycle: list, unused_nodes: set,
                                      unused_nodes_list: list,
                                      unused_nodes_index_dict, i_1):
    best_move: tuple = tuple()
    best_cost_delta = np.iinfo(distance_matrix.dtype).max

    sorted_indices = np.argsort(distance_matrix[cycle[i_1]])
    nearest_five_indices = set()

    for node in sorted_indices:
        if node in unused_nodes:
            nearest_five_indices.add(unused_nodes_index_dict[node])
        if len(nearest_five_indices) >= 5:
            break

    for i_2 in nearest_five_indices:
        move, cost_delta = _find_outside_swap_move(distance_matrix, cycle, unused_nodes_list, i_1, i_2)
        if cost_delta < best_cost_delta:
            best_cost_delta = cost_delta
            best_move = move
    return best_move, best_cost_delta


def _find_outside_swap_move(distance_matrix: np.ndarray, cycle: list, unused_nodes: list, i_1, i_2):
    node_1_0 = cycle[i_1 - 1]
    node_1_1 = cycle[i_1]
    node_1_2 = cycle[i_1 + 1]
    edge_length_1_1 = distance_matrix[node_1_0, node_1_1]
    edge_length_1_2 = distance_matrix[node_1_1, node_1_2]

    node_2 = unused_nodes[i_2]
    new_edge_length_1 = distance_matrix[node_1_0, node_2]
    new_edge_length_2 = distance_matrix[node_2, node_1_2]

    current_cost = edge_length_1_1 + edge_length_1_2
    new_cost = new_edge_length_1 + new_edge_length_2
    cost_delta = new_cost - current_cost
    move = (i_1, i_2)
    return move, cost_delta


def _find_edge_swap_move(distance_matrix: np.ndarray, cycle: list, i_1, i_2):
    node_1_1 = cycle[i_1]
    node_1_2 = cycle[i_1 + 1]
    edge_length_1 = distance_matrix[node_1_1, node_1_2]

    node_2_1 = cycle[i_2]
    node_2_2 = cycle[i_2 + 1]
    edge_length_2 = distance_matrix[node_2_1, node_2_2]

    new_edge_length_1 = distance_matrix[node_1_1, node_2_1]
    new_edge_length_2 = distance_matrix[node_1_2, node_2_2]

    current_cost = edge_length_1 + edge_length_2
    new_cost = new_edge_length_1 + new_edge_length_2
    cost_delta = new_cost - current_cost
    move = (i_1 + 1, i_2)
    return move, cost_delta


class LocalSearchOperation(Enum):
    swap_outside = "SO"
    swap_inside = "SI"
    swap_edges = "SE"


class CandidateSteepSearch(SearchProblemSolver):
    def solve(self, distance_matrix: np.ndarray, start_cycle: List[int] = None, max_time=None) -> List[int]:
        if start_cycle:
            cycle = start_cycle
        else:
            cycle = RandomSearch(0.0).solve(distance_matrix)

        unused_nodes_list = [node for node in range(distance_matrix.shape[0]) if node not in cycle]

        while True:
            unused_nodes = set(unused_nodes_list)

            move, operation, cost_delta = CandidateSteepSearch.__find_best_move(distance_matrix, cycle, unused_nodes,
                                                                                unused_nodes_list)

            if cost_delta >= 0:
                break

            i_1, i_2 = move
            if operation == LocalSearchOperation.swap_outside:
                cycle[i_1], unused_nodes_list[i_2] = unused_nodes_list[i_2], cycle[i_1]
                if i_1 == 0:
                    cycle[-1] = cycle[0]
            elif operation == LocalSearchOperation.swap_edges:
                cycle[i_1:i_2 + 1] = reversed(cycle[i_1:i_2 + 1])

        return cycle

    @staticmethod
    def __find_best_move(distance_matrix: np.ndarray, cycle: list, unused_nodes: set, unused_nodes_list: list):
        best_cost_delta = np.iinfo(distance_matrix.dtype).max
        best_move: tuple = tuple()
        best_operation = None

        unused_nodes_index_dict = {}
        for i, node_index in enumerate(unused_nodes):
            unused_nodes_index_dict[node_index] = i

        for i_1 in range(len(cycle) - 1):
            move, cost_delta = _find_best_outside_swap_candidate(distance_matrix, cycle, unused_nodes,
                                                                 unused_nodes_list, unused_nodes_index_dict, i_1)
            if cost_delta < best_cost_delta:
                best_cost_delta = cost_delta
                best_move = move
                best_operation = LocalSearchOperation.swap_outside

            if i_1 >= len(cycle) - 2:
                continue

            move, cost_delta = CandidateSteepSearch.__find_best_edge_swap_move(distance_matrix, cycle, i_1)
            if cost_delta < best_cost_delta:
                best_cost_delta = cost_delta
                best_move = move
                best_operation = LocalSearchOperation.swap_edges

        return best_move, best_operation, best_cost_delta

    @staticmethod
    def __find_best_edge_swap_move(distance_matrix: np.ndarray, cycle: list, i_1):
        best_cost_delta = np.iinfo(distance_matrix.dtype).max
        best_move: tuple = tuple()
        for i_2 in range(i_1 + 2, len(cycle) - 1):
            move, cost_delta = _find_edge_swap_move(distance_matrix, cycle, i_1, i_2)
            if cost_delta < best_cost_delta:
                best_cost_delta = cost_delta
                best_move = move
        return best_move, best_cost_delta


class ScoreSteepSearch(SearchProblemSolver):
    def __init__(self):
        self.cache = set()

    def solve(self, distance_matrix: np.ndarray, start_cycle=None, max_time=None) -> List[int]:
        cycle = RandomSearch(0.0).solve(distance_matrix)
        unused_nodes_list = [node for node in range(distance_matrix.shape[0]) if node not in cycle]

        while True:
            move, operation, cost_delta = self.__find_best_move(distance_matrix, cycle, unused_nodes_list)

            if cost_delta >= 0:
                break

            i_1, i_2 = move
            if operation == LocalSearchOperation.swap_outside:
                self.cache.add((unused_nodes_list[i_2], cycle[i_1], operation))
                cycle[i_1], unused_nodes_list[i_2] = unused_nodes_list[i_2], cycle[i_1]
                if i_1 == 0:
                    cycle[-1] = cycle[0]
            elif operation == LocalSearchOperation.swap_edges:
                self.cache.add((cycle[i_1], cycle[i_2], operation))
                cycle[i_1:i_2 + 1] = reversed(cycle[i_1:i_2 + 1])
        return cycle

    def __find_best_move(self, distance_matrix: np.ndarray, cycle: list, unused_nodes_list: list):
        best_cost_delta = np.iinfo(distance_matrix.dtype).max
        best_move: tuple = tuple()
        best_operation = None

        unused_nodes_index_dict = {}
        for i in range(len(unused_nodes_list)):
            unused_nodes_index_dict[unused_nodes_list[i]] = i

        for i_1 in range(len(cycle) - 1):
            move, cost_delta = _find_best_outside_swap(distance_matrix, cycle, unused_nodes_list, i_1)
            if cost_delta < best_cost_delta:
                best_cost_delta = cost_delta
                best_move = move
                best_operation = LocalSearchOperation.swap_outside

            if i_1 >= len(cycle) - 2:
                continue

            move, cost_delta = self.__find_best_edge_swap_move(distance_matrix, cycle, i_1)
            if cost_delta < best_cost_delta:
                best_cost_delta = cost_delta
                best_move = move
                best_operation = LocalSearchOperation.swap_edges

        return best_move, best_operation, best_cost_delta

    def __find_best_edge_swap_move(self, distance_matrix: np.ndarray, cycle: list, i_1):
        best_cost_delta = np.iinfo(distance_matrix.dtype).max
        best_move: tuple = tuple()
        for i_2 in range(i_1 + 2, len(cycle) - 1):
            if (cycle[i_1 + 1], cycle[i_2 + 1], LocalSearchOperation.swap_edges) in self.cache:
                continue

            move = (i_1 + 1, i_2)
            cost_delta = self.__find_edge_swap_move(distance_matrix, cycle, i_1, i_2)
            if cost_delta < best_cost_delta:
                best_cost_delta = cost_delta
                best_move = move
        return best_move, best_cost_delta

    @staticmethod
    def __find_edge_swap_move(distance_matrix: np.ndarray, cycle: list, i_1, i_2):
        node_1_1 = cycle[i_1]
        node_1_2 = cycle[i_1 + 1]
        edge_length_1 = distance_matrix[node_1_1, node_1_2]

        node_2_1 = cycle[i_2]
        node_2_2 = cycle[i_2 + 1]
        edge_length_2 = distance_matrix[node_2_1, node_2_2]

        new_edge_length_1 = distance_matrix[node_1_1, node_2_1]
        new_edge_length_2 = distance_matrix[node_1_2, node_2_2]

        current_cost = edge_length_1 + edge_length_2
        new_cost = new_edge_length_1 + new_edge_length_2
        cost_delta = new_cost - current_cost
        return cost_delta

    def __find_best_outside_swap(self, distance_matrix: np.ndarray, cycle: list, unused_nodes_list: list, i_1):
        best_move: tuple = tuple()
        best_cost_delta = np.iinfo(distance_matrix.dtype).max

        for i_2 in range(len(unused_nodes_list)):
            if (cycle[i_1], unused_nodes_list[i_2], LocalSearchOperation.swap_outside) in self.cache:
                continue

            move = (i_1, i_2)
            cost_delta = self.__find_outside_swap_move(distance_matrix, cycle, unused_nodes_list, i_1, i_2)
            if cost_delta < best_cost_delta:
                best_cost_delta = cost_delta
                best_move = move
        return best_move, best_cost_delta

    @staticmethod
    def __find_outside_swap_move(distance_matrix: np.ndarray, cycle: list, unused_nodes: list, i_1, i_2):
        node_1_0 = cycle[i_1 - 1]
        node_1_1 = cycle[i_1]
        node_1_2 = cycle[i_1 + 1]
        edge_length_1_1 = distance_matrix[node_1_0, node_1_1]
        edge_length_1_2 = distance_matrix[node_1_1, node_1_2]

        node_2 = unused_nodes[i_2]
        new_edge_length_1 = distance_matrix[node_1_0, node_2]
        new_edge_length_2 = distance_matrix[node_2, node_1_2]

        current_cost = edge_length_1_1 + edge_length_1_2
        new_cost = new_edge_length_1 + new_edge_length_2
        cost_delta = new_cost - current_cost
        return cost_delta
