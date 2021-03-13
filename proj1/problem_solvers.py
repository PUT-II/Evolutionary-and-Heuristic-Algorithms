from abc import ABC, abstractmethod
from typing import List

import numpy as np


def _calculate_path_length(distance_matrix: np.ndarray, path: List[int]) -> np.uint64:
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


class ProblemSolver(ABC):

    @abstractmethod
    def solve(self, distance_matrix: np.ndarray, start_node: int = 1) -> List[int]:
        """ Creates path for given graph.

        :param distance_matrix: distance_matrix of given graph
        :param start_node: index of path start node
        :return: Path generated by given implementation of problem solver
        """
        pass


class NearestNeighbourProblemSolver(ProblemSolver):
    def solve(self, distance_matrix: np.ndarray, start_node: int = 0) -> List[int]:
        shape = _validate_shape(distance_matrix.shape)

        # Copy distance to not modify argument matrix
        distance_matrix_ = distance_matrix.copy()

        current_node_index = start_node

        # Loop until path contains 50% of nodes
        path = [start_node]
        while len(path) < round(shape[0] / 2):
            # Get node closest to current node
            current_node_distances = distance_matrix_[current_node_index, :]
            closest_node_index = np.argmin(current_node_distances)

            # Mark edges as already used
            distance_matrix_[current_node_index, :] = -1
            distance_matrix_[:, current_node_index] = -1

            # Add node to path
            path.append(closest_node_index)
            # Set added node as current node
            current_node_index = closest_node_index

        # Close path to make Hamiltonian cycle
        result_cycle = path + [path[0]]

        return result_cycle


class GreedyCycleProblemSolver(ProblemSolver):

    def solve(self, distance_matrix: np.ndarray, start_node: int = 0) -> List[int]:
        shape = _validate_shape(distance_matrix.shape)

        # Loop until path contains 50% of nodes
        path: List[int] = [start_node, np.argmin(distance_matrix[start_node, :])]
        remaining_indices: List[int] = [index for index in list(range(shape[0])) if index not in path]
        while len(path) < round(shape[0] / 2):
            best_candidate_index = -1
            best_candidate_position = -1
            best_length = np.uint64(-1)

            # Loop on remaining indices and find best candidate
            temp_cycle = path + [path[0]]
            cycle_length = _calculate_path_length(distance_matrix, temp_cycle)
            cycle_length = np.uint64(cycle_length)
            for candidate_index in remaining_indices:
                # Loop on positions of temporal cycle
                for i in range(1, len(temp_cycle)):
                    # Calculate all edge lengths
                    index_1 = temp_cycle[i - 1]
                    index_2 = temp_cycle[i]
                    candidate_length_1 = distance_matrix[index_1][candidate_index]
                    candidate_length_2 = distance_matrix[index_2][candidate_index]
                    edge_length = distance_matrix[index_1][index_2]

                    # Length when candidate is put in i-th position
                    new_length = cycle_length + candidate_length_1 + candidate_length_2 - edge_length

                    if new_length < best_length:
                        best_candidate_index = candidate_index
                        best_candidate_position = i
                        best_length = new_length

            # Insert best candidate in best position to path
            path.insert(best_candidate_position, best_candidate_index)
            remaining_indices.remove(best_candidate_index)

        # Close path to make Hamiltonian cycle
        result_cycle = path + [path[0]]

        return result_cycle


class RegretCycleProblemSolver(ProblemSolver):

    def solve(self, distance_matrix: np.ndarray, start_node: int = 0) -> List[int]:
        shape = _validate_shape(distance_matrix.shape)

        # Loop until path contains 50% of nodes
        path: List[int] = [start_node, np.argmin(distance_matrix[start_node, :])]
        remaining_indices: List[int] = [index for index in list(range(shape[0])) if index not in path]
        while len(path) < round(shape[0] / 2):

            # Loop on remaining indices and find best candidates
            temp_path = path + [path[0]]
            k_regret_candidates = []
            for candidate_index in remaining_indices:
                # Loop on positions of temporal cycle
                k_regret_list = []
                for i in range(1, len(temp_path)):
                    # Calculate all edge lengths
                    index_1 = temp_path[i - 1]
                    index_2 = temp_path[i]
                    candidate_length_1 = distance_matrix[index_1][candidate_index]
                    candidate_length_2 = distance_matrix[index_2][candidate_index]
                    edge_length = distance_matrix[index_1][index_2]

                    # Calculate cost of putting candidate in i-th position
                    if len(path) <= 2:
                        cost = np.int64(candidate_length_1) + candidate_length_2
                    else:
                        cost = np.int64(candidate_length_1) + candidate_length_2 - edge_length
                    k_regret_list.append((candidate_index, i, cost))

                # Get (up to) 3 best candidate positions
                k_regret_list = sorted(k_regret_list, key=lambda elem: elem[2])[0:3]

                # Calculate k_regret for 1-st best candidate
                if len(path) <= 2:
                    k_regret = k_regret_list[0][2] - k_regret_list[1][2]
                else:
                    k_regret = 2 * k_regret_list[0][2] - k_regret_list[1][2] - k_regret_list[2][2]
                k_regret_candidates.append((k_regret_list[0][0], k_regret_list[0][1], k_regret))

            # Choose candidate with highest regret
            best_index, best_position, _ = max(k_regret_candidates, key=lambda elem: elem[2])
            path.insert(best_position, best_index)
            remaining_indices.remove(best_index)

        # Close path to make Hamiltonian cycle
        result_cycle = path + [path[0]]

        return result_cycle
