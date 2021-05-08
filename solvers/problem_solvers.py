from typing import List

import numpy as np

import common.utils as utils
from common.interfaces import ProblemSolver


class NearestNeighbourProblemSolver(ProblemSolver):
    def solve(self, distance_matrix: np.ndarray, start_node: int = 0, _=None) -> List[int]:
        shape = utils.validate_shape(distance_matrix.shape)

        max_distance = np.iinfo(distance_matrix.dtype).max

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
            distance_matrix_[current_node_index, :] = max_distance
            distance_matrix_[:, current_node_index] = max_distance

            # Add node to path
            path.append(closest_node_index)
            # Set added node as current node
            current_node_index = closest_node_index

        # Close path to make Hamiltonian cycle
        result_cycle = path + [path[0]]

        return result_cycle


class GreedyCycleProblemSolver(ProblemSolver):

    def solve(self, distance_matrix: np.ndarray, start_node: int = 0, start_cycle: List[int] = None) -> List[int]:
        shape = utils.validate_shape(distance_matrix.shape)

        # Loop until path contains 50% of nodes
        if start_cycle:
            path: List[int] = start_cycle[:-1]
        else:
            path: List[int] = [start_node, np.argmin(distance_matrix[start_node, :])]
        remaining_indices: List[int] = [index for index in list(range(shape[0])) if index not in path]
        while len(path) < round(shape[0] / 2):
            best_candidate_index = -1
            best_candidate_position = -1
            best_cost = np.uint64(-1)

            # Loop on remaining indices and find best candidate
            temp_cycle = path + [path[0]]
            for candidate_index in remaining_indices:
                # Loop on positions of temporal cycle
                for i in range(1, len(temp_cycle)):
                    # Calculate all edge lengths
                    index_1 = temp_cycle[i - 1]
                    index_2 = temp_cycle[i]
                    candidate_length_1 = distance_matrix[index_1, candidate_index]
                    candidate_length_2 = distance_matrix[index_2, candidate_index]
                    edge_length = distance_matrix[index_1, index_2]

                    # Calculate cost of putting candidate in i-th position
                    if len(path) <= 2:
                        cost = np.int64(candidate_length_1) + candidate_length_2
                    else:
                        cost = np.int64(candidate_length_1) + candidate_length_2 - edge_length

                    if cost < best_cost:
                        best_candidate_index = candidate_index
                        best_candidate_position = i
                        best_cost = cost

            # Insert best candidate in best position to path
            path.insert(best_candidate_position, best_candidate_index)
            remaining_indices.remove(best_candidate_index)

        # Close path to make Hamiltonian cycle
        result_cycle = path + [path[0]]

        return result_cycle


class RegretCycleProblemSolver(ProblemSolver):

    def solve(self, distance_matrix: np.ndarray, start_node: int = 0, _=None) -> List[int]:
        shape = utils.validate_shape(distance_matrix.shape)

        # Loop until path contains 50% of nodes
        path: List[int] = [start_node, np.argmin(distance_matrix[start_node, :])]
        remaining_indices: List[int] = [index for index in list(range(shape[0])) if index not in path]
        while len(path) < round(shape[0] / 2):

            # Loop on remaining indices and find best candidates
            temp_cycle = path + [path[0]]
            k_regret_candidates = []
            for candidate_index in remaining_indices:
                # Loop on positions of temporal cycle
                cost_list = []
                for i in range(1, len(temp_cycle)):
                    # Calculate all edge lengths
                    index_1 = temp_cycle[i - 1]
                    index_2 = temp_cycle[i]
                    candidate_length_1 = distance_matrix[index_1, candidate_index]
                    candidate_length_2 = distance_matrix[index_2, candidate_index]
                    edge_length = distance_matrix[index_1, index_2]

                    # Calculate cost of putting candidate in i-th position
                    if len(path) <= 2:
                        cost = np.int64(candidate_length_1) + candidate_length_2
                    else:
                        cost = np.int64(candidate_length_1) + candidate_length_2 - edge_length
                    cost_list.append((candidate_index, i, cost))

                # Get (up to) 3 best candidate positions
                cost_list = sorted(cost_list, key=lambda elem: elem[2])[0:3]

                # Calculate k_regret for 1-st best candidate
                if len(path) <= 2:
                    k_regret = cost_list[0][2] - cost_list[1][2]
                else:
                    k_regret = 2 * cost_list[0][2] - cost_list[1][2] - cost_list[2][2]
                k_regret_candidates.append((cost_list[0][0], cost_list[0][1], k_regret))

            # Choose candidate with highest regret
            best_index, best_position, _ = max(k_regret_candidates, key=lambda elem: elem[2])
            path.insert(best_position, best_index)
            remaining_indices.remove(best_index)

        # Close path to make Hamiltonian cycle
        result_cycle = path + [path[0]]

        return result_cycle
