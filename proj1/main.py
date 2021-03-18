from typing import Set

import numpy as np
from tsplib95.models import StandardProblem

import common.utils as utils
from common.interfaces import ProblemSolver
from proj1.problem_solvers import GreedyCycleProblemSolver, NearestNeighbourProblemSolver, RegretCycleProblemSolver

_EXPERIMENT_COUNT: int = 50


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
    distance_matrix: np.ndarray = utils.create_distance_matrix(problem)
    paths = []
    for node_index in random_nodes:
        path = problem_solver.solve(distance_matrix, node_index - 1)
        paths.append(path)

    # Calculate min, max and average cycle lengths
    cycle_lengths = [utils.calculate_path_length(distance_matrix, path) for path in paths]
    minimum_length, shortest_cycle_index = min((val, idx) for (idx, val) in enumerate(cycle_lengths))
    maximum_length = max(cycle_lengths)
    average_length = round(sum(cycle_lengths) / len(cycle_lengths))

    # Draw best cycle
    shortest_path = [index + 1 for index in paths[shortest_cycle_index]]
    result_title = f"{result_title}_{shortest_path[0]}"
    utils.draw_graph(problem, shortest_path, result_title, minimum_length)

    print(result_title)
    print(f"Cycle length (min) : {minimum_length}")
    print(f"Cycle length (max) : {maximum_length}")
    print(f"Cycle length (avg) : {average_length}")
    print()


def main():
    import shutil
    import tsplib95

    problem_a: StandardProblem = tsplib95.load('data/kroa100.tsp')
    problem_b: StandardProblem = tsplib95.load('data/krob100.tsp')

    shutil.rmtree("graphs/", ignore_errors=True)

    run_experiment(problem_a, NearestNeighbourProblemSolver(), "kroa100_nn")
    run_experiment(problem_b, NearestNeighbourProblemSolver(), "krob100_nn")
    run_experiment(problem_a, GreedyCycleProblemSolver(), "kroa100_gc")
    run_experiment(problem_b, GreedyCycleProblemSolver(), "krob100_gc")
    run_experiment(problem_a, RegretCycleProblemSolver(), "kroa100_rc")
    run_experiment(problem_b, RegretCycleProblemSolver(), "krob100_rc")


if __name__ == '__main__':
    main()
