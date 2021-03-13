from typing import List, Set

import numpy as np
from tsplib95.models import StandardProblem

from proj1.problem_solvers import ProblemSolver, GreedyCycleProblemSolver, NearestNeighbourProblemSolver, \
    RegretCycleProblemSolver

_EXPERIMENT_COUNT: int = 50


def _calculate_distance(point_1: List[int], point_2: List[int]) -> int:
    """ Calculates distance between given points.

    :param point_1: point coordinates (x, y)
    :param point_2: point coordinates (x, y)
    :return: distance between given points
    """
    pow_x: int = (point_1[0] - point_2[0]) ** 2
    pow_y: int = (point_1[1] - point_2[1]) ** 2

    return round((pow_x + pow_y) ** 0.5)


def _create_distance_matrix(problem: StandardProblem) -> np.ndarray:
    """ Creates distance matrix for given graph.

    :param problem: problem which contains graph nodes
    :return: distance matrix
    """
    matrix = np.full(shape=(problem.dimension, problem.dimension), dtype=np.uint32, fill_value=-1)

    node_dict = dict(problem.node_coords)

    for node_index_1 in range(problem.dimension - 1):
        node_1 = node_dict[node_index_1 + 1]
        for node_index_2 in range(node_index_1 + 1, problem.dimension):
            node_2 = node_dict[node_index_2 + 1]
            distance = _calculate_distance(node_1, node_2)
            matrix[node_index_1, node_index_2] = distance
            matrix[node_index_2, node_index_1] = distance

    return matrix


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

    _, ax = plt.subplots()
    nx.draw_networkx_edges(graph, pos, edge_color='red')
    nx.draw_networkx_nodes(graph, pos, node_size=1, node_color=node_colors, ax=ax)
    nx.draw_networkx_labels(graph, pos, font_size=5)

    if not os.path.exists('./graphs/'):
        os.makedirs('./graphs/')

    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
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

    if problem.dimension < _EXPERIMENT_COUNT:
        raise ValueError(f"problem.dimension < {_EXPERIMENT_COUNT}")

    # Pick different random nodes
    random_nodes: Set[int] = set()
    while len(random_nodes) < _EXPERIMENT_COUNT:
        random_node = random.randint(1, problem.dimension)

        if random_node not in random_nodes:
            random_nodes.add(random_node)

    # Create distance matrix and solve TSP problem using every random node
    distance_matrix: np.ndarray = _create_distance_matrix(problem)
    paths = []
    for node_index in random_nodes:
        path = problem_solver.solve(distance_matrix, node_index - 1)
        paths.append(path)

    # Calculate min, max and average cycle lengths
    cycle_lengths = [_calculate_path_length(distance_matrix, path) for path in paths]
    minimum_length, shortest_cycle_index = min((val, idx) for (idx, val) in enumerate(cycle_lengths))
    maximum_length = max(cycle_lengths)
    average_length = round(sum(cycle_lengths) / len(cycle_lengths))

    # Draw best cycle
    shortest_path = [index + 1 for index in paths[shortest_cycle_index]]
    result_title = f"{result_title}_{shortest_path[0]}"
    _draw_graph(problem, shortest_path, result_title, minimum_length)

    print(result_title)
    print(f"Cycle length (min) : {minimum_length}")
    print(f"Cycle length (max) : {maximum_length}")
    print(f"Cycle length (avg) : {average_length}")
    print()


def main():
    import shutil
    import tsplib95

    problem_a: StandardProblem = tsplib95.load('./data/kroa100.tsp')
    problem_b: StandardProblem = tsplib95.load('./data/krob100.tsp')

    shutil.rmtree("./graphs/", ignore_errors=True)

    run_experiment(problem_a, NearestNeighbourProblemSolver(), "kroa100_nn")
    run_experiment(problem_b, NearestNeighbourProblemSolver(), "krob100_nn")
    run_experiment(problem_a, GreedyCycleProblemSolver(), "kroa100_gc")
    run_experiment(problem_b, GreedyCycleProblemSolver(), "krob100_gc")
    run_experiment(problem_a, RegretCycleProblemSolver(), "kroa100_rc")
    run_experiment(problem_b, RegretCycleProblemSolver(), "krob100_rc")


if __name__ == '__main__':
    main()
