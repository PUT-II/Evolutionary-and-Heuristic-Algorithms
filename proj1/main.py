from typing import List, Set

import numpy as np
from tsplib95.models import StandardProblem

from proj1.classes.GreedyCycleProblemSolver import GreedyCycleProblemSolver
from proj1.classes.NearestNeighbourProblemSolver import NearestNeighbourProblemSolver
from proj1.classes.ProblemSolver import ProblemSolver

_EXPERIMENT_COUNT: int = 50


def _draw_graph(problem: StandardProblem, path: List[int], result_title: str, path_length: int):
    import os
    import matplotlib.pyplot as plt
    import networkx as nx

    start_node = path[0]
    graph = nx.Graph()

    node_coords_dict = dict(problem.node_coords)
    node_colors = []
    for node_index in node_coords_dict:
        graph.add_node(node_index, pos=tuple(node_coords_dict[node_index]))
        node_colors.append('red' if node_index == start_node else 'blue')

    for i in range(len(path) - 1):
        graph.add_edge(path[i], path[i + 1])

    pos = nx.get_node_attributes(graph, 'pos')

    nx.draw_networkx_edges(graph, pos, edge_color='red')
    nx.draw_networkx_nodes(graph, pos, node_size=1, node_color=node_colors)
    nx.draw_networkx_labels(graph, pos, font_size=5)

    if not os.path.exists('./graphs/'):
        os.makedirs('./graphs/')

    plt.suptitle(result_title)
    plt.title(f"Length : {path_length}")
    plt.savefig(f"./graphs/{result_title}.pdf")
    plt.show()
    plt.clf()


def _calculate_path_length(distance_matrix: np.ndarray, path: list) -> int:
    total_length = 0
    for i in range(len(path) - 1):
        total_length += distance_matrix[path[i], path[i + 1]]

    return total_length


def run_experiment(problem: StandardProblem, problem_solver: ProblemSolver, result_title: str = "graph"):
    """ Solves problem using 50 different randomly selected start nodes.

    :param problem: problem which contains graph nodes
    :param problem_solver: specific implementation of ProblemSolver
    :param result_title: title which will be given to result image
    """
    import random
    global _EXPERIMENT_COUNT

    random_nodes: Set[int] = set()

    if problem.dimension < _EXPERIMENT_COUNT:
        raise ValueError("problem.dimension < 50")

    while len(random_nodes) < _EXPERIMENT_COUNT:
        random_node = random.randint(1, problem.dimension)

        if random_node not in random_nodes:
            random_nodes.add(random_node)

    distance_matrix: np.ndarray = ProblemSolver.create_distance_matrix(problem)
    paths = []
    for node_index in random_nodes:
        path = problem_solver.solve(distance_matrix, node_index - 1)
        paths.append(path)

    path_lengths = [_calculate_path_length(distance_matrix, path) for path in paths]
    shortest_length, shortest_path_index = min((val, idx) for (idx, val) in enumerate(path_lengths))

    shortest_path = [index + 1 for index in paths[shortest_path_index]]

    result_title = f"{result_title}_{shortest_path[0]}"
    _draw_graph(problem, shortest_path, result_title, shortest_length)
    print(result_title)
    print(f"Path length : {shortest_length}")
    print()


def main():
    import tsplib95

    problem_a: StandardProblem = tsplib95.load('./data/kroa100.tsp')
    problem_b: StandardProblem = tsplib95.load('./data/krob100.tsp')

    run_experiment(problem_a, NearestNeighbourProblemSolver(), "kroa100_nn")
    run_experiment(problem_b, NearestNeighbourProblemSolver(), "krob100_nn")
    run_experiment(problem_a, GreedyCycleProblemSolver(), "krob100_gc")
    run_experiment(problem_b, GreedyCycleProblemSolver(), "krob100_gc")


if __name__ == '__main__':
    main()
