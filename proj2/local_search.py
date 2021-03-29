import random
from enum import Enum
from typing import List

import numpy as np

import common.utils as utils
from common.interfaces import SearchProblemSolver


def _find_best_outside_swap(distance_matrix: np.ndarray, cycle: list, unused_nodes: list, i_1):
    node_1_0 = cycle[i_1 - 1]
    node_1_1 = cycle[i_1]
    node_1_2 = cycle[i_1 + 1]
    edge_length_1_1 = distance_matrix[node_1_0, node_1_1]
    edge_length_1_2 = distance_matrix[node_1_1, node_1_2]

    # Swap outside
    current_cost_1 = edge_length_1_1 + edge_length_1_2

    best_move: tuple = tuple()
    best_cost_delta = np.iinfo(np.int32).max
    for i_2 in range(len(unused_nodes)):
        node_2 = unused_nodes[i_2]

        new_edge_length_1 = distance_matrix[node_1_0, node_2]
        new_edge_length_2 = distance_matrix[node_2, node_1_2]

        new_cost_1 = new_edge_length_1 + new_edge_length_2

        cost_delta_1 = new_cost_1 - current_cost_1
        if cost_delta_1 < best_cost_delta:
            best_cost_delta = cost_delta_1
            best_move = (i_1, i_2)
    return best_move, best_cost_delta


class LocalSearchOperation(Enum):
    swap_outside = "SO"
    swap_inside = "SI"
    swap_edges = "SE"


class RandomSearch(SearchProblemSolver):
    def solve(self, distance_matrix: np.ndarray) -> List[int]:
        shape = utils.validate_shape(distance_matrix.shape)

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
    def solve(self, distance_matrix: np.ndarray) -> List[int]:
        cycle = RandomSearch().solve(distance_matrix)

        unused_nodes = [node for node in range(distance_matrix.shape[0]) if node not in cycle]

        while True:
            move, operation, cost_delta = NodeSwapSteepSearch.__find_best_move(distance_matrix, cycle, unused_nodes)

            if cost_delta >= 0:
                break

            i_1, i_2 = move
            # cycle_copy = cycle.copy()
            if operation == LocalSearchOperation.swap_outside:
                cycle[i_1], unused_nodes[i_2] = unused_nodes[i_2], cycle[i_1]
                if i_1 == 0:
                    cycle[-1] = cycle[0]
            elif operation == LocalSearchOperation.swap_inside:
                cycle[i_1], cycle[i_2] = cycle[i_2], cycle[i_1]
                if i_1 == 0:
                    cycle[-1] = cycle[0]

            # path_length = utils.calculate_path_length(distance_matrix, cycle + [cycle[0]])
            # path_copy_length = utils.calculate_path_length(distance_matrix, cycle_copy + [cycle_copy[0]])
            # if cost_delta != path_copy_length - path_length:
            #     raise ArithmeticError("Incorrect cost delta")
            # cycle = cycle_copy

        return cycle

    @staticmethod
    def __find_best_move(distance_matrix: np.ndarray, cycle: list, unused_nodes: list):
        best_cost_delta = np.iinfo(np.int32).max
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

            # if i_1 >= len(cycle) - 2:
            #     continue
            #
            # for i_2 in range(i_1 + 1, len(cycle)):
            #     node_2_0 = cycle[i_2 - 1]
            #     node_2_1 = cycle[i_2]
            #     if i_2 != len(cycle) - 1:
            #         node_2_2 = cycle[i_2 + 1]
            #     else:
            #         node_2_2 = cycle[0]
            #     edge_length_2_1 = distance_matrix[node_2_0, node_2_1]
            #     edge_length_2_2 = distance_matrix[node_2_1, node_2_2]
            #
            #     # Swap nodes
            #     new_edge_length_1_1 = distance_matrix[node_1_0, node_2_1]
            #     new_edge_length_1_2 = distance_matrix[node_2_1, node_1_2]
            #     new_edge_length_2_1 = distance_matrix[node_2_0, node_1_1]
            #     new_edge_length_2_2 = distance_matrix[node_1_1, node_2_2]
            #
            #     new_cost_3 = new_edge_length_1_1 + new_edge_length_1_2 + new_edge_length_2_1 + new_edge_length_2_2
            #     current_cost_3 = current_cost_1 + edge_length_2_1 + edge_length_2_2
            #     if i_1 == i_2 - 1:
            #         new_cost_3 += edge_length_2_1
            #         current_cost_3 -= edge_length_2_1
            #
            #     cost_delta_3 = new_cost_3 - current_cost_3
            #
            #     if cost_delta_3 < best_cost_delta:
            #         best_cost_delta = cost_delta_3
            #         best_move = (i_1, i_2)
            #         best_operation = LocalSearchOperation.swap_inside

        return best_move, best_operation, best_cost_delta

    @staticmethod
    def __find_best_node_swap_move(distance_matrix: np.ndarray, cycle: list, i_1):
        best_cost_delta = np.iinfo(np.int32).max
        best_move: tuple = tuple()

        if i_1 != 0:
            node_1_0 = cycle[i_1 - 1]
        else:
            node_1_0 = cycle[i_1 - 2]

        node_1_1 = cycle[i_1]
        node_1_2 = cycle[i_1 + 1]
        edge_length_1_1 = distance_matrix[node_1_0, node_1_1]
        edge_length_1_2 = distance_matrix[node_1_1, node_1_2]

        current_cost_1 = edge_length_1_1 + edge_length_1_2

        # Iterate over nodes after node at i_1
        for i_2 in range(i_1 + 1, len(cycle) - 1):
            # Node before node at i_2
            node_2_0 = cycle[i_2 - 1]
            node_2_1 = cycle[i_2]
            # Node after node at i_2
            node_2_2 = cycle[i_2 + 1]

            # Edge lengths for i_2 before swap
            edge_length_2_1 = distance_matrix[node_2_0, node_2_1]
            edge_length_2_2 = distance_matrix[node_2_1, node_2_2]

            # Every new edge length after swap
            new_edge_length_1_1 = distance_matrix[node_1_0, node_2_1]
            new_edge_length_1_2 = distance_matrix[node_2_1, node_1_2]
            new_edge_length_2_1 = distance_matrix[node_2_0, node_1_1]
            new_edge_length_2_2 = distance_matrix[node_1_1, node_2_2]

            cost = current_cost_1 + edge_length_2_1 + edge_length_2_2
            new_cost = new_edge_length_1_1 + new_edge_length_1_2 + new_edge_length_2_1 + new_edge_length_2_2
            if i_1 == i_2 - 1:
                cost -= edge_length_1_2
                new_cost += edge_length_1_2
            elif i_1 == 0 and i_2 == len(cycle) - 2:
                cost -= edge_length_2_2
                new_cost += edge_length_2_2

            cost_delta = new_cost - cost

            if cost_delta < best_cost_delta:
                best_cost_delta = cost_delta
                best_move = (i_1, i_2)
        return best_move, best_cost_delta


class EdgeSwapSteepSearch(SearchProblemSolver):
    def solve(self, distance_matrix: np.ndarray) -> List[int]:
        cycle = RandomSearch().solve(distance_matrix)
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
        best_cost_delta = np.iinfo(np.int32).max
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
        node_1_1 = cycle[i_1]
        node_1_2 = cycle[i_1 + 1]
        edge_length_1 = distance_matrix[node_1_1, node_1_2]

        best_cost_delta = np.iinfo(np.int32).max
        best_move: tuple = tuple()
        for i_2 in range(i_1 + 2, len(cycle) - 1):
            node_2_1 = cycle[i_2]
            node_2_2 = cycle[i_2 + 1]
            edge_length_2 = distance_matrix[node_2_1, node_2_2]

            new_edge_length_1 = distance_matrix[node_1_1, node_2_1]
            new_edge_length_2 = distance_matrix[node_1_2, node_2_2]

            current_cost = edge_length_1 + edge_length_2
            new_cost = new_edge_length_1 + new_edge_length_2
            cost_delta = new_cost - current_cost
            if cost_delta < best_cost_delta:
                best_cost_delta = cost_delta
                best_move = (i_1 + 1, i_2)
        return best_move, best_cost_delta


class GreedyLocalSearch(SearchProblemSolver):

    def __init__(self, use_node_swap: bool = False, use_edge_swap: bool = False):
        self.operations = (LocalSearchOperation.swap_outside,)
        if use_node_swap:
            self.operations += (LocalSearchOperation.swap_inside,)

        if use_edge_swap:
            self.operations += (LocalSearchOperation.swap_edges,)

    def solve(self, distance_matrix: np.ndarray) -> List[int]:
        shape = utils.validate_shape(distance_matrix.shape)
        cycle = RandomSearch().solve(distance_matrix)

        unused_nodes = [i for i in range(shape[0]) if i not in cycle]

        while True:
            move, operation, cost_delta = self.__find_good_enough_move(distance_matrix, cycle, unused_nodes)

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
            elif operation == LocalSearchOperation.swap_edges:
                cycle[i_1:i_2 + 1] = reversed(cycle[i_1:i_2 + 1])

        result_cycle = cycle + [cycle[0]]
        return result_cycle

    def __find_good_enough_move(self, distance_matrix: np.ndarray, cycle: list, unused_nodes: list):
        best_cost_delta = np.iinfo(np.int32).max
        best_move: tuple = tuple()
        best_operation = None

        start_index = random.choice(range(len(cycle)))
        # True = Right, False = Left
        search_direction = random.choice((False, True))

        loop_start = start_index if search_direction else 0
        loop_end = len(cycle) - 1 if search_direction else start_index
        index_range = range(loop_start, loop_end)

        if not search_direction:
            index_range = reversed(index_range)

        # print("Right" if search_direction else "Left")
        for i in index_range:
            random_operation = random.choice(self.operations)

            if random_operation == LocalSearchOperation.swap_edges:
                move, cost_delta = GreedyLocalSearch.__find_good_enough_edge_swap_move(distance_matrix, cycle, i,
                                                                                       search_direction)
                if cost_delta < best_cost_delta:
                    best_move = move
                    best_cost_delta = cost_delta
                    best_operation = LocalSearchOperation.swap_edges
            elif random_operation == LocalSearchOperation.swap_inside:
                move, cost_delta = GreedyLocalSearch.__find_good_enough_node_swap_move(distance_matrix, cycle, i,
                                                                                       search_direction)
                if cost_delta < best_cost_delta:
                    best_move = move
                    best_cost_delta = cost_delta
                    best_operation = LocalSearchOperation.swap_inside
            elif random_operation == LocalSearchOperation.swap_outside:
                move, cost_delta = GreedyLocalSearch.__find_good_enough_outside_swap(distance_matrix, cycle,
                                                                                     unused_nodes, i, search_direction)
                if cost_delta < best_cost_delta:
                    best_move = move
                    best_cost_delta = cost_delta
                    best_operation = LocalSearchOperation.swap_outside

        return best_move, best_operation, best_cost_delta

    @staticmethod
    def __find_good_enough_node_swap_move(distance_matrix: np.ndarray, cycle: list, i_1, search_direction: bool):
        best_cost_delta = np.iinfo(np.int32).max
        best_move: tuple = tuple()

        if i_1 != 0:
            node_1_0 = cycle[i_1 - 1]
        else:
            node_1_0 = cycle[i_1 - 2]

        node_1_1 = cycle[i_1]
        node_1_2 = cycle[i_1 + 1]
        edge_length_1_1 = distance_matrix[node_1_0, node_1_1]
        edge_length_1_2 = distance_matrix[node_1_1, node_1_2]

        current_cost_1 = edge_length_1_1 + edge_length_1_2

        # Loop variables
        loop_start = i_1 + 1 if search_direction else 0
        loop_end = len(cycle) - 1 if search_direction else i_1
        index_range = range(loop_start, loop_end)
        if not search_direction:
            index_range = reversed(index_range)

        for i_2 in index_range:
            # Node before node at i_2
            node_2_0 = cycle[i_2 - 1]
            node_2_1 = cycle[i_2]
            # Node after node at i_2
            node_2_2 = cycle[i_2 + 1]

            # Edge lengths for i_2 before swap
            edge_length_2_1 = distance_matrix[node_2_0, node_2_1]
            edge_length_2_2 = distance_matrix[node_2_1, node_2_2]

            # Every new edge length after swap
            new_edge_length_1_1 = distance_matrix[node_1_0, node_2_1]
            new_edge_length_1_2 = distance_matrix[node_2_1, node_1_2]
            new_edge_length_2_1 = distance_matrix[node_2_0, node_1_1]
            new_edge_length_2_2 = distance_matrix[node_1_1, node_2_2]

            cost = current_cost_1 + edge_length_2_1 + edge_length_2_2
            new_cost = new_edge_length_1_1 + new_edge_length_1_2 + new_edge_length_2_1 + new_edge_length_2_2
            if i_1 == i_2 - 1:
                cost -= edge_length_1_2
                new_cost += edge_length_1_2
            elif i_1 == 0 and i_2 == len(cycle) - 2:
                cost -= edge_length_2_2
                new_cost += edge_length_2_2

            cost_delta = new_cost - cost

            if cost_delta < best_cost_delta:
                best_cost_delta = cost_delta
                best_move = (i_1, i_2)
        return best_move, best_cost_delta

    @staticmethod
    def __find_good_enough_edge_swap_move(distance_matrix: np.ndarray, cycle: list, i_1, search_direction: bool):
        best_cost_delta = np.iinfo(np.int32).max
        best_move: tuple = tuple()

        node_1_1 = cycle[i_1]
        node_1_2 = cycle[i_1 + 1]
        edge_length_1 = distance_matrix[node_1_1, node_1_2]

        # Loop variables
        loop_start = i_1 + 1 if search_direction else 0
        loop_end = len(cycle) - 1 if search_direction else i_1
        index_range = range(loop_start, loop_end)
        if not search_direction:
            index_range = reversed(index_range)

        for i_2 in index_range:
            node_2_1 = cycle[i_2]
            node_2_2 = cycle[i_2 + 1]
            edge_length_2 = distance_matrix[node_2_1, node_2_2]

            new_edge_length_1 = distance_matrix[node_1_1, node_2_1]
            new_edge_length_2 = distance_matrix[node_1_2, node_2_2]

            current_cost = edge_length_1 + edge_length_2
            new_cost = new_edge_length_1 + new_edge_length_2
            cost_delta = new_cost - current_cost
            if cost_delta < best_cost_delta:
                best_cost_delta = cost_delta
                best_move = (i_1 + 1, i_2)
        return best_move, best_cost_delta

    @staticmethod
    def __find_good_enough_outside_swap(distance_matrix: np.ndarray, cycle: list, unused_nodes: list, i_1,
                                        search_direction: bool):
        node_1_0 = cycle[i_1 - 1]
        node_1_1 = cycle[i_1]
        node_1_2 = cycle[i_1 + 1]
        edge_length_1_1 = distance_matrix[node_1_0, node_1_1]
        edge_length_1_2 = distance_matrix[node_1_1, node_1_2]

        # Swap outside
        current_cost_1 = edge_length_1_1 + edge_length_1_2

        best_move: tuple = tuple()
        best_cost_delta = np.iinfo(np.int32).max

        # Loop variables
        loop_start = i_1 + 1 if search_direction else 0
        loop_end = len(cycle) - 1 if search_direction else i_1
        index_range = range(loop_start, loop_end)
        if not search_direction:
            index_range = reversed(index_range)

        for i_2 in index_range:
            node_2 = unused_nodes[i_2]

            new_edge_length_1 = distance_matrix[node_1_0, node_2]
            new_edge_length_2 = distance_matrix[node_2, node_1_2]

            new_cost_1 = new_edge_length_1 + new_edge_length_2

            cost_delta_1 = new_cost_1 - current_cost_1
            if cost_delta_1 < best_cost_delta:
                best_cost_delta = cost_delta_1
                best_move = (i_1, i_2)
        return best_move, best_cost_delta
