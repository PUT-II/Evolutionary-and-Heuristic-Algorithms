from typing import List, Set

from tsplib95.models import StandardProblem

from proj1.NearestNeighbourProblemSolver import NearestNeighbourProblemSolver
from proj1.ProblemSolver import ProblemSolver


def _draw_graph(problem: StandardProblem, path: List[int], file_name: str) -> None:
    import matplotlib.pyplot as plt
    import networkx as nx

    graph = nx.Graph()

    node_coords_dict = dict(problem.node_coords)
    for node_index in node_coords_dict:
        graph.add_node(node_index, pos=tuple(node_coords_dict[node_index]))

    for i in range(len(path) - 1):
        graph.add_edge(path[i], path[i + 1])

    pos = nx.get_node_attributes(graph, 'pos')

    nx.draw(graph, pos, node_size=1, edge_color='r', with_labels=True, font_size=5)
    plt.savefig(file_name)
    plt.clf()


def solve_problem(problem: StandardProblem,
                  problem_solver: ProblemSolver,
                  file_name: str = "graph",
                  start_node: int = 1):
    import time
    import os

    time_start = time.time()
    path = problem_solver.solve(problem, start_node)
    print(f"{file_name} time : {round(time.time() - time_start, 4)}s")

    if not os.path.exists('./graphs/'):
        os.makedirs('./graphs/')

    _draw_graph(problem, path, f'./graphs/{file_name}.pdf')


def run_experiment(problem: StandardProblem, problem_solver: ProblemSolver, file_name: str = "graph"):
    import random

    used_nodes: Set[int] = set()

    while len(used_nodes) < 50:
        random_node = random.randint(1, problem.dimension)

        if random_node in used_nodes:
            continue

        solve_problem(problem, problem_solver, f"{file_name}_{random_node}", random_node)
        used_nodes.add(random_node)


def _main():
    import tsplib95

    problem_a: StandardProblem = tsplib95.load('./proj1/kroa100.tsp')
    problem_b: StandardProblem = tsplib95.load('./proj1/krob100.tsp')

    run_experiment(problem_a, NearestNeighbourProblemSolver(), "kroa100_nn")
    run_experiment(problem_b, NearestNeighbourProblemSolver(), "krob100_nn")


if __name__ == '__main__':
    _main()
