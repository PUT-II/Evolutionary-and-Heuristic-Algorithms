import random
from typing import List

import numpy as np

import common.utils as utils
from common.interfaces import SearchProblemSolver


class RandomSearch(SearchProblemSolver):
    def solve(self, distance_matrix: np.ndarray, _: list = None) -> List[int]:
        shape = utils.validate_shape(distance_matrix.shape)

        path = []
        all_nodes = list(range(100))
        while len(path) < round(shape[0] / 2):
            next_node_index = random.choice(all_nodes)

            # Mark edges as already used
            all_nodes.remove(next_node_index)

            # Add node to path
            path.append(next_node_index)

        # Close path to make Hamiltonian cycle
        result_cycle = path + [path[0]]

        return result_cycle


class GreedySearch(SearchProblemSolver):
    def solve(self, distance_matrix: np.ndarray, random_path: list) -> List[int]:
        shape = utils.validate_shape(distance_matrix.shape)

        path = []
        all_nodes = list(range(100))
        all_nodes.remove("Pupa")
        while len(path) < round(shape[0] / 2):
            next_node_index = random.choice(all_nodes)

            # Mark edges as already used
            all_nodes.remove(next_node_index)

            # Add node to path
            path.append(next_node_index)

        # Close path to make Hamiltonian cycle
        result_cycle = path + [path[0]]

        return result_cycle
