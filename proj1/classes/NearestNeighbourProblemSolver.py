from typing import List, Dict

from tsplib95.models import StandardProblem

from proj1.classes.ProblemSolver import ProblemSolver


class NearestNeighbourProblemSolver(ProblemSolver):
    def solve(self, problem: StandardProblem, start_node: int = 1) -> List[int]:
        node_coords_dict: Dict[int, List[int]] = dict(problem.node_coords)

        selected_node: List[int] = node_coords_dict[start_node]
        del node_coords_dict[start_node]

        closed_path = [start_node]
        while len(closed_path) < round(problem.dimension / 2):
            nearest_node: tuple = (0, float("inf"))
            for node_index in node_coords_dict:
                node = problem.node_coords[node_index]
                node_distance = ProblemSolver.calculate_distance(selected_node, node)
                if node_distance < nearest_node[1]:
                    nearest_node = (node_index, node_distance)

            nearest_node_index = nearest_node[0]
            closed_path.append(nearest_node_index)

            selected_node = node_coords_dict[nearest_node_index]
            del node_coords_dict[nearest_node_index]

        closed_path.append(start_node)
        return closed_path
