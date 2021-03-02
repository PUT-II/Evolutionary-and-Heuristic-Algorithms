from typing import List

from tsplib95.models import StandardProblem

from proj1.ProblemSolver import ProblemSolver


class NearestNeighbourProblemSolver(ProblemSolver):
    def solve(self, problem: StandardProblem, start_node: int = 1) -> List[int]:
        import operator

        node_coords_dict = dict(problem.node_coords)

        selected_node = node_coords_dict[start_node]
        del node_coords_dict[start_node]

        closed_path = [start_node]
        while len(closed_path) < round(problem.dimension / 2):
            node_distances = {}
            for node_index in node_coords_dict:
                node = problem.node_coords[node_index]
                node_distances[node_index] = ProblemSolver.calculate_distance(selected_node, node)

            nearest_node_index = min(node_distances.items(), key=operator.itemgetter(1))[0]
            closed_path.append(nearest_node_index)

            selected_node = node_coords_dict[nearest_node_index]
            del node_coords_dict[nearest_node_index]

        closed_path.append(start_node)
        return closed_path
