from typing import List

import numpy as np

from proj1.classes.ProblemSolver import ProblemSolver


class NearestNeighbourProblemSolver(ProblemSolver):
    def solve(self, distance_matrix: np.ndarray, start_node: int = 0) -> List[int]:
        shape = distance_matrix.shape
        if len(shape) != 2 or shape[0] != shape[1]:
            raise ValueError('Wrong distance matrix shape: len(shape) != 2 or shape[0] != shape[1]')

        current_node_index = start_node

        closed_path = [start_node]
        while len(closed_path) < round(shape[0] / 2):
            current_node_distances = distance_matrix[current_node_index, :]
            closest_node_index = np.argmin(current_node_distances)

            distance_matrix[current_node_index, :] = -1
            distance_matrix[:, current_node_index] = -1

            closed_path.append(closest_node_index)
            current_node_index = closest_node_index

        closed_path.append(start_node)

        return closed_path
