import time
from typing import Set

import numpy as np
from tsplib95.models import StandardProblem

import common.utils as utils
from common.interfaces import SearchProblemSolver
from solvers.local_search import RandomSearch, EdgeSwapSteepSearch, NodeSwapSteepSearch, GreedyLocalSearch

_EXPERIMENT_COUNT: int = 100


def run_experiment(problem: StandardProblem, problem_solver: SearchProblemSolver, result_title: str = "graph") -> float:
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
    times = []
    for _ in random_nodes:
        time_start = time.time()
        path = problem_solver.solve(distance_matrix)
        time_end = time.time()
        times.append((time_end - time_start))
        paths.append(path)

    # Calculate min, max and average cycle lengths
    cycle_lengths = [utils.calculate_path_length(distance_matrix, path) for path in paths]
    minimum_length, shortest_cycle_index = min((val, idx) for (idx, val) in enumerate(cycle_lengths))
    maximum_length = max(cycle_lengths)
    average_length = round(sum(cycle_lengths) / len(cycle_lengths))

    maximum_time = max(times)
    minimum_time = min(times)
    average_time = round(sum(times) / len(times), 3)

    # Draw best cycle
    shortest_path = [index + 1 for index in paths[shortest_cycle_index]]
    result_title = f"{result_title}_{shortest_path[0]}"
    utils.draw_graph(problem, shortest_path, result_title, minimum_length)

    print(result_title)
    print(f"Cycle length (min) : {minimum_length}")
    print(f"Cycle length (max) : {maximum_length}")
    print(f"Cycle length (avg) : {average_length}")
    print(f"Time (min) : {round(minimum_time * 1000.0)}ms")
    print(f"Time (max) : {round(maximum_time * 1000.0)}ms")
    print(f"Time (avg) : {round(average_time * 1000.0)}ms")
    print()
    return average_time


def main():
    import shutil
    import tsplib95

    problem_a: StandardProblem = tsplib95.load('data/kroa100.tsp')
    problem_b: StandardProblem = tsplib95.load('data/krob100.tsp')

    shutil.rmtree("./graphs/", ignore_errors=True)

    average_time = []
    average_time.append(run_experiment(problem_a, NodeSwapSteepSearch(), "kroa100_nsss"))
    average_time.append(run_experiment(problem_a, EdgeSwapSteepSearch(), "kroa100_esss"))
    average_time.append(run_experiment(problem_a, GreedyLocalSearch(use_node_swap=True), "kroa100_gls_ns"))
    average_time.append(run_experiment(problem_a, GreedyLocalSearch(use_edge_swap=True), "kroa100_gls_es"))
    average_time.append(run_experiment(problem_b, NodeSwapSteepSearch(), "krob100_nsss"))
    average_time.append(run_experiment(problem_b, EdgeSwapSteepSearch(), "krob100_esss"))
    average_time.append(run_experiment(problem_b, GreedyLocalSearch(use_node_swap=True), "krob100_gls_ns"))
    average_time.append(run_experiment(problem_b, GreedyLocalSearch(use_edge_swap=True), "krob100_gls_es"))
    max_avg_time = max(average_time)
    print(f"Max average time : {round(max_avg_time * 1000.0)}")
    print()
    run_experiment(problem_a, RandomSearch(max_avg_time), "kroa100_rs")
    run_experiment(problem_b, RandomSearch(max_avg_time), "krob100_rs")


if __name__ == '__main__':
    main()
