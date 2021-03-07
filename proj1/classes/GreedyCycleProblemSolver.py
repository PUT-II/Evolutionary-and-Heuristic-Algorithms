from typing import List

import numpy as np

from proj1.classes.ProblemSolver import ProblemSolver


class GreedyCycleProblemSolver(ProblemSolver):
    def solve(self, distance_matrix: np.ndarray, start_node: int = 0) -> List[int]:
        shape = distance_matrix.shape
        if len(shape) != 2 or shape[0] != shape[1]:
            raise ValueError('Wrong distance matrix shape: len(shape) != 2 or shape[0] != shape[1]')

        distance_matrix_ = distance_matrix.copy()

        previous_node_index = start_node
        current_node_index = np.argmin(distance_matrix_[previous_node_index, :])

        distance_matrix_[previous_node_index, current_node_index] = -1
        distance_matrix_[current_node_index, previous_node_index] = -1

        closed_path = [start_node, current_node_index]
        while len(closed_path) < round(shape[0] / 2):
            previous_node_distances = distance_matrix_[previous_node_index, :]
            current_node_distances = distance_matrix_[current_node_index, :]
            sum_of_distances = previous_node_distances + current_node_distances

            closest_node_index = np.argmin(sum_of_distances)

            distance_matrix_[previous_node_index, :] = -1
            distance_matrix_[:, previous_node_index] = -1
            distance_matrix_[closest_node_index, current_node_index] = -1
            distance_matrix_[current_node_index, closest_node_index] = -1

            closed_path.append(closest_node_index)
            previous_node_index = current_node_index
            current_node_index = closest_node_index

        closed_path.append(start_node)

        return closed_path
