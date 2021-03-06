from typing import List, Set

import numpy as np
from tsplib95.models import StandardProblem

from proj1.classes.NearestNeighbourProblemSolver import NearestNeighbourProblemSolver
from proj1.classes.ProblemSolver import ProblemSolver

_EXPERIMENT_COUNT: int = 50


def _draw_graph(problem: StandardProblem, path: List[int], file_name: str, start_node: int = -1) -> None:
    import matplotlib.pyplot as plt
    import networkx as nx

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
    plt.savefig(file_name)
    plt.clf()


def solve_problem(problem: StandardProblem,
                  problem_solver: ProblemSolver,
                  file_name: str = "graph",
                  start_node: int = 1):
    """ Uses given ProblemSolver to determine a path for given problem.

    :param problem: problem which contains graph nodes
    :param problem_solver: specific implementation of ProblemSolver
    :param file_name: name of file to which result graph will be saved
    :param start_node: index of path start node
    """
    import time
    import os

    time_start = time.time()
    distance_matrix: np.ndarray = ProblemSolver.create_distance_matrix(problem)
    path = problem_solver.solve(distance_matrix, start_node - 1)
    print(f"{file_name} time : {round(time.time() - time_start, 4)}s")

    if not os.path.exists('./graphs/'):
        os.makedirs('./graphs/')

    _draw_graph(problem, [index + 1 for index in path], f'./graphs/{file_name}.pdf', start_node)


def run_experiment(problem: StandardProblem, problem_solver: ProblemSolver, file_name: str = "graph"):
    """ Solves problem using 50 different randomly selected start nodes.

    :param problem: problem which contains graph nodes
    :param problem_solver: specific implementation of ProblemSolver
    :param file_name: name of file to which result graph will be saved
    """
    import random
    global _EXPERIMENT_COUNT

    random_nodes: Set[int] = set()

    if problem.dimension < _EXPERIMENT_COUNT:
        raise ValueError("problem.dimension < 50")

    while len(random_nodes) < _EXPERIMENT_COUNT:
        random_node = random.randint(0, problem.dimension)

        if random_node not in random_nodes:
            random_nodes.add(random_node)

    for node_index in random_nodes:
        solve_problem(problem, problem_solver, f"{file_name}_{node_index}", node_index)


def main():
    import tsplib95

    problem_a: StandardProblem = tsplib95.load('./data/kroa100.tsp')
    problem_b: StandardProblem = tsplib95.load('./data/krob100.tsp')

    run_experiment(problem_a, NearestNeighbourProblemSolver(), "kroa100_nn")
    run_experiment(problem_b, NearestNeighbourProblemSolver(), "krob100_nn")


if __name__ == '__main__':
    main()
