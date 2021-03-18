import random
from typing import List

import numpy as np

from common.interfaces import ProblemSolver


def _calculate_length(distance_matrix: np.ndarray, path: List[int]) -> np.uint64:
    length = np.uint64(0)
    for i in range(len(path) - 1):
        index_1 = path[i]
        index_2 = path[i + 1]
        length += distance_matrix[index_1][index_2]

    return length


def _validate_shape(shape: tuple) -> tuple:
    if len(shape) != 2 or shape[0] != shape[1]:
        raise ValueError('Wrong distance matrix shape: len(shape) != 2 or shape[0] != shape[1]')
    return shape


class RandomSearch(ProblemSolver):
    def solve(self, distance_matrix: np.ndarray, start_node: int = 0) -> List[int]:
        shape = _validate_shape(distance_matrix.shape)

        path = [start_node]
        all_nodes = list(range(100))
        all_nodes.remove(start_node)
        while len(path) < round(shape[0] / 2):
            next_node_index = random.choice(all_nodes)

            # Mark edges as already used
            all_nodes.remove(next_node_index)

            # Add node to path
            path.append(next_node_index)

        # Close path to make Hamiltonian cycle
        result_cycle = path + [path[0]]

        return result_cycle
