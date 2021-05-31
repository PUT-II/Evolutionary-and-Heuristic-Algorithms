import random
import time
from enum import Enum
from typing import List

import numpy as np

import common.utils as utils
from common.interfaces import SearchProblemSolver


def _find_best_outside_swap(distance_matrix: np.ndarray, cycle: list, unused_nodes: list, i_1):
    best_move: tuple = tuple()
    best_cost_delta = np.iinfo(distance_matrix.dtype).max
    for i_2 in range(len(unused_nodes)):
        move, cost_delta = _find_outside_swap_move(distance_matrix, cycle, unused_nodes, i_1, i_2)
        if cost_delta < best_cost_delta:
            best_cost_delta = cost_delta
            best_move = move
    return best_move, best_cost_delta


def _find_outside_swap_move(distance_matrix: np.ndarray, cycle: list, unused_nodes: list, index_1, index_2):
    node_1_0 = cycle[index_1 - 1]
    node_1_1 = cycle[index_1]
    node_1_2 = cycle[index_1 + 1]
    edge_length_1_1 = distance_matrix[node_1_0, node_1_1]
    edge_length_1_2 = distance_matrix[node_1_1, node_1_2]

    node_2 = unused_nodes[index_2]
    new_edge_length_1 = distance_matrix[node_1_0, node_2]
    new_edge_length_2 = distance_matrix[node_2, node_1_2]

    current_cost = edge_length_1_1 + edge_length_1_2
    new_cost = new_edge_length_1 + new_edge_length_2
    cost_delta = new_cost - current_cost
    move = (index_1, index_2)
    return move, cost_delta


def _find_node_swap_move(distance_matrix: np.ndarray, cycle: list, index_1, index_2):
    if index_1 != 0:
        node_1_0 = cycle[index_1 - 1]
    else:
        node_1_0 = cycle[index_1 - 2]

    node_1_1 = cycle[index_1]
    node_1_2 = cycle[index_1 + 1]
    edge_length_1_1 = distance_matrix[node_1_0, node_1_1]
    edge_length_1_2 = distance_matrix[node_1_1, node_1_2]

    # Node before node at i_2
    node_2_0 = cycle[index_2 - 1]
    node_2_1 = cycle[index_2]
    # Node after node at i_2
    if index_2 < len(cycle) - 1:
        node_2_2 = cycle[index_2 + 1]
    else:
        node_2_2 = cycle[1]

    # Edge lengths for i_2 before swap
    edge_length_2_1 = distance_matrix[node_2_0, node_2_1]
    edge_length_2_2 = distance_matrix[node_2_1, node_2_2]

    # Every new edge length after swap
    new_edge_length_1_1 = distance_matrix[node_1_0, node_2_1]
    new_edge_length_1_2 = distance_matrix[node_2_1, node_1_2]
    new_edge_length_2_1 = distance_matrix[node_2_0, node_1_1]
    new_edge_length_2_2 = distance_matrix[node_1_1, node_2_2]

    current_cost = edge_length_1_1 + edge_length_1_2
    cost = current_cost + edge_length_2_1 + edge_length_2_2
    new_cost = new_edge_length_1_1 + new_edge_length_1_2 + new_edge_length_2_1 + new_edge_length_2_2
    if index_1 == index_2 - 1:
        cost -= edge_length_1_2
        new_cost += edge_length_1_2
    elif index_1 == 0 and index_2 == len(cycle) - 2:
        cost -= edge_length_2_2
        new_cost += edge_length_2_2

    cost_delta = new_cost - cost
    move = (index_1, index_2)
    return move, cost_delta


def _find_edge_swap_move(distance_matrix: np.ndarray, cycle: list, index_1, index_2):
    node_1_1 = cycle[index_1]
    node_1_2 = cycle[index_1 + 1]
    edge_length_1 = distance_matrix[node_1_1, node_1_2]

    node_2_1 = cycle[index_2]
    node_2_2 = cycle[index_2 + 1]
    edge_length_2 = distance_matrix[node_2_1, node_2_2]

    new_edge_length_1 = distance_matrix[node_1_1, node_2_1]
    new_edge_length_2 = distance_matrix[node_1_2, node_2_2]

    current_cost = edge_length_1 + edge_length_2
    new_cost = new_edge_length_1 + new_edge_length_2
    cost_delta = new_cost - current_cost
    move = (index_1 + 1, index_2)
    return move, cost_delta


class LocalSearchOperation(Enum):
    swap_outside = "SO"
    swap_inside = "SI"
    swap_edges = "SE"


class RandomSearch(SearchProblemSolver):
    def __init__(self, gen_time: float = 0.0):
        self.gen_time = gen_time

    def solve(self, distance_matrix: np.ndarray, start_cycle=None) -> List[int]:
        if self.gen_time == 0.0:
            return RandomSearch.__generate_random_cycle(distance_matrix)

        best_result_cycle = []
        best_result_cycle_length = np.iinfo(distance_matrix.dtype).max
        time_start = time.time()
        while time.time() - time_start < self.gen_time:
            result_cycle = RandomSearch.__generate_random_cycle(distance_matrix)
            result_cycle_length = utils.calculate_path_length(distance_matrix, result_cycle)
            if result_cycle_length < best_result_cycle_length:
                best_result_cycle_length = result_cycle_length
                best_result_cycle = result_cycle

        return best_result_cycle

    @staticmethod
    def __generate_random_cycle(distance_matrix: np.ndarray):
        shape = distance_matrix.shape
        path = []
        all_nodes = list(range(shape[0]))
        while len(path) < round(shape[0] / 2):
            next_node_index = random.choice(all_nodes)

            # Mark node as already used
            all_nodes.remove(next_node_index)

            # Add node to path
            path.append(next_node_index)

        # Close path to make Hamiltonian cycle
        result_cycle = path + [path[0]]
        return result_cycle


class NodeSwapSteepSearch(SearchProblemSolver):
    def solve(self, distance_matrix: np.ndarray, start_cycle=None) -> List[int]:
        cycle = RandomSearch(0.0).solve(distance_matrix)

        unused_nodes = [node for node in range(distance_matrix.shape[0]) if node not in cycle]

        while True:
            move, operation, cost_delta = NodeSwapSteepSearch.__find_best_move(distance_matrix, cycle, unused_nodes)

            if cost_delta >= 0:
                break

            i_1, i_2 = move
            if operation == LocalSearchOperation.swap_outside:
                cycle[i_1], unused_nodes[i_2] = unused_nodes[i_2], cycle[i_1]
                if i_1 == 0:
                    cycle[-1] = cycle[0]
            elif operation == LocalSearchOperation.swap_inside:
                cycle[i_1], cycle[i_2] = cycle[i_2], cycle[i_1]
                if i_1 == 0:
                    cycle[-1] = cycle[0]

        return cycle

    @staticmethod
    def __find_best_move(distance_matrix: np.ndarray, cycle: list, unused_nodes: list):
        best_cost_delta = np.iinfo(distance_matrix.dtype).max
        best_move: tuple = tuple()
        best_operation = None
        for i_1 in range(len(cycle) - 1):
            # Swap outside
            move, cost_delta = _find_best_outside_swap(distance_matrix, cycle, unused_nodes, i_1)
            if cost_delta < best_cost_delta:
                best_cost_delta = cost_delta
                best_move = move
                best_operation = LocalSearchOperation.swap_outside

            move, cost_delta = NodeSwapSteepSearch.__find_best_node_swap_move(distance_matrix, cycle, i_1)
            if cost_delta < best_cost_delta:
                best_cost_delta = cost_delta
                best_move = move
                best_operation = LocalSearchOperation.swap_inside

        return best_move, best_operation, best_cost_delta

    @staticmethod
    def __find_best_node_swap_move(distance_matrix: np.ndarray, cycle: list, i_1):
        best_cost_delta = np.iinfo(distance_matrix.dtype).max
        best_move: tuple = tuple()

        # Iterate over nodes after node at i_1
        for i_2 in range(i_1 + 1, len(cycle) - 1):
            move, cost_delta = _find_node_swap_move(distance_matrix, cycle, i_1, i_2)

            if cost_delta < best_cost_delta:
                best_cost_delta = cost_delta
                best_move = (i_1, i_2)
        return best_move, best_cost_delta


class EdgeSwapSteepSearch(SearchProblemSolver):
    def solve(self, distance_matrix: np.ndarray, start_cycle=None) -> List[int]:
        cycle = RandomSearch(0.0).solve(distance_matrix)
        unused_nodes = [node for node in range(distance_matrix.shape[0]) if node not in cycle]

        while True:
            move, operation, cost_delta = EdgeSwapSteepSearch.__find_best_move(distance_matrix, cycle, unused_nodes)

            if cost_delta >= 0:
                break

            i_1, i_2 = move
            if operation == LocalSearchOperation.swap_outside:
                cycle[i_1], unused_nodes[i_2] = unused_nodes[i_2], cycle[i_1]
                if i_1 == 0:
                    cycle[-1] = cycle[0]
            elif operation == LocalSearchOperation.swap_edges:
                cycle[i_1:i_2 + 1] = reversed(cycle[i_1:i_2 + 1])

        return cycle

    @staticmethod
    def __find_best_move(distance_matrix: np.ndarray, cycle: list, unused_nodes: list):
        best_cost_delta = np.iinfo(distance_matrix.dtype).max
        best_move: tuple = tuple()
        best_operation = None
        for i_1 in range(len(cycle) - 1):
            move, cost_delta = _find_best_outside_swap(distance_matrix, cycle, unused_nodes, i_1)
            if cost_delta < best_cost_delta:
                best_cost_delta = cost_delta
                best_move = move
                best_operation = LocalSearchOperation.swap_outside

            if i_1 >= len(cycle) - 2:
                continue

            move, cost_delta = EdgeSwapSteepSearch.__find_best_edge_swap_move(distance_matrix, cycle, i_1)
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


class GreedyLocalSearch(SearchProblemSolver):

    def __init__(self, use_node_swap: bool = False, use_edge_swap: bool = False):
        self.operations = (LocalSearchOperation.swap_outside,)
        if use_node_swap:
            self.operations += (LocalSearchOperation.swap_inside,)

        if use_edge_swap:
            self.operations += (LocalSearchOperation.swap_edges,)

    def solve(self, distance_matrix: np.ndarray, start_cycle=None) -> List[int]:
        shape = utils.validate_shape(distance_matrix.shape)
        cycle = RandomSearch(0.0).solve(distance_matrix)

        unused_nodes = [i for i in range(shape[0]) if i not in cycle]

        while True:
            move, operation, cost_delta = self.__find_first_favorable_move(distance_matrix, cycle, unused_nodes)

            if cost_delta == 0:
                continue
            elif cost_delta > 0:
                break

            i_1, i_2 = move
            cycle_copy = cycle.copy()
            if operation == LocalSearchOperation.swap_outside:
                cycle_copy[i_1], unused_nodes[i_2] = unused_nodes[i_2], cycle_copy[i_1]
                if i_1 == 0:
                    cycle_copy[-1] = cycle_copy[0]
            elif operation == LocalSearchOperation.swap_inside:
                cycle_copy[i_1], cycle_copy[i_2] = cycle_copy[i_2], cycle_copy[i_1]
                if i_1 == 0:
                    cycle_copy[-1] = cycle_copy[0]
            elif operation == LocalSearchOperation.swap_edges:
                cycle_copy[i_1:i_2 + 1] = reversed(cycle_copy[i_1:i_2 + 1])

            cycle_length = utils.calculate_path_length(distance_matrix, cycle)
            cycle_copy_length = utils.calculate_path_length(distance_matrix, cycle_copy)
            if cost_delta != cycle_copy_length - cycle_length:
                raise ArithmeticError("Incorrect cost delta")
            cycle = cycle_copy

        result_cycle = cycle + [cycle[0]]
        return result_cycle

    def __find_first_favorable_move(self, distance_matrix: np.ndarray, cycle: list, unused_nodes: list):
        cycle_indices = list(range(len(cycle)))
        random_indices_1 = list(np.random.permutation(cycle_indices))
        random_indices_1.remove(len(cycle) - 1)

        for i_1 in random_indices_1:
            random_operations = list(np.random.permutation(self.operations))

            for random_operation in random_operations:
                if random_operation == LocalSearchOperation.swap_outside:
                    random_indices_2 = list(np.random.permutation(range(len(unused_nodes))))
                    move, cost_delta, operation = \
                        GreedyLocalSearch.__find_first_favorable_outside_swap(distance_matrix, cycle, unused_nodes, i_1,
                                                                              random_indices_2)
                else:
                    if i_1 >= len(cycle) - 1:
                        continue

                    random_indices_2 = list(np.random.permutation(cycle_indices))
                    random_indices_2.remove(i_1)

                    if len(cycle) - 1 in random_indices_2:
                        random_indices_2.remove(len(cycle) - 1)

                    if random_operation == LocalSearchOperation.swap_edges:
                        move, cost_delta, operation = \
                            GreedyLocalSearch.__find_first_favorable_edge_swap_move(distance_matrix, cycle, i_1,
                                                                                    random_indices_2)
                    elif random_operation == LocalSearchOperation.swap_inside:
                        move, cost_delta, operation = \
                            GreedyLocalSearch.__find_first_favorable_node_swap_move(distance_matrix, cycle, i_1,
                                                                                    random_indices_2)
                    else:
                        continue

                if cost_delta < 0:
                    return move, operation, cost_delta

        return tuple(), None, np.iinfo(distance_matrix.dtype).max

    @staticmethod
    def __find_first_favorable_node_swap_move(distance_matrix: np.ndarray, cycle: list, index_1, indices: list):
        for index_2 in indices:
            i_1 = index_1 if index_2 > index_1 else index_2
            i_2 = index_2 if index_2 > index_1 else index_1

            move, cost_delta = _find_node_swap_move(distance_matrix, cycle, i_1, i_2)
            if cost_delta < 0:
                return move, cost_delta, LocalSearchOperation.swap_inside
        return tuple(), np.iinfo(distance_matrix.dtype).max, LocalSearchOperation.swap_inside

    @staticmethod
    def __find_first_favorable_edge_swap_move(distance_matrix: np.ndarray, cycle: list, i, random_indices_2: list):

        for random_index in random_indices_2:
            i_1 = i if random_index > i else random_index
            i_2 = random_index if random_index > i else i

            if i_2 < i_1 + 2:
                continue

            move, cost_delta = _find_edge_swap_move(distance_matrix, cycle, i_1, i_2)
            if cost_delta < np.iinfo(distance_matrix.dtype).max:
                return move, cost_delta, LocalSearchOperation.swap_edges
        return tuple(), np.iinfo(distance_matrix.dtype).max, LocalSearchOperation.swap_edges

    @staticmethod
    def __find_first_favorable_outside_swap(distance_matrix: np.ndarray, cycle: list, unused_nodes: list, i,
                                            random_indices_2: list):
        for random_index in random_indices_2:
            i_1 = i if random_index > i else random_index
            i_2 = random_index if random_index < i else i

            move, cost_delta = _find_outside_swap_move(distance_matrix, cycle, unused_nodes, i_1, i_2)
            if cost_delta < np.iinfo(distance_matrix.dtype).max:
                return move, cost_delta, LocalSearchOperation.swap_outside
        return tuple(), np.iinfo(distance_matrix.dtype).max, LocalSearchOperation.swap_outside
