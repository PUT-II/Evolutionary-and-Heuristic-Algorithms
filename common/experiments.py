import random
import time
from typing import Set

import numpy as np
from tsplib95.models import StandardProblem

from common import utils
from common.interfaces import ProblemSolver, SearchProblemSolver, IteratedSearchProblemSolver


def __process_results(problem: StandardProblem,
                      result_title: str,
                      paths: list,
                      times: list,
                      search_invocations: list = None):
    distance_matrix: np.ndarray = utils.create_distance_matrix(problem)

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
    print(f"Path : {shortest_path}")
    print(f"Cycle length (min) : {minimum_length}")
    print(f"Cycle length (max) : {maximum_length}")
    print(f"Cycle length (avg) : {average_length}")
    print(f"Time (min) : {round(minimum_time * 1000.0)}ms")
    print(f"Time (max) : {round(maximum_time * 1000.0)}ms")
    print(f"Time (avg) : {round(average_time * 1000.0)}ms")

    if search_invocations:
        maximum_invocations = max(search_invocations)
        minimum_invocations = min(search_invocations)
        average_invocations = round(sum(search_invocations) / len(search_invocations), 3)
        print(f"Search invocations (min) : {maximum_invocations}")
        print(f"Search invocations (max) : {minimum_invocations}")
        print(f"Search invocations (avg) : {average_invocations}")

    print()
    return average_time


def run_experiment_constructive(problem: StandardProblem,
                                problem_solver: ProblemSolver,
                                result_title: str = "graph",
                                experiment_count: int = 50):
    """ Solves problem using 50 different randomly selected start nodes.

    :param problem: problem which contains graph nodes
    :param problem_solver: specific implementation of ProblemSolver
    :param result_title: title which will be given to result image
    """

    if problem.dimension < experiment_count:
        raise ValueError(f"problem.dimension < {experiment_count}")

    # Pick different random nodes
    random_nodes: Set[int] = set()
    while len(random_nodes) < experiment_count:
        random_node = random.randint(1, problem.dimension)

        if random_node not in random_nodes:
            random_nodes.add(random_node)

    # Create distance matrix and solve TSP problem using every random start node
    distance_matrix: np.ndarray = utils.create_distance_matrix(problem)
    for i in range(len(distance_matrix)):
        distance_matrix[i, i] = np.iinfo(distance_matrix.dtype).max

    paths = []
    times = []
    for node_index in random_nodes:
        time_start = time.time()
        path = problem_solver.solve(distance_matrix, node_index - 1)
        time_end = time.time()
        times.append((time_end - time_start))
        paths.append(path)

    __process_results(problem, result_title, paths, times)


def run_experiment_local_search(problem: StandardProblem,
                                problem_solver: SearchProblemSolver,
                                result_title: str = "graph",
                                experiment_count: int = 100) -> float:
    """ Solves problem using 50 different randomly selected start nodes.

    :param problem: problem which contains graph nodes
    :param problem_solver: specific implementation of ProblemSolver
    :param result_title: title which will be given to result image
    """
    if problem.dimension < experiment_count:
        raise ValueError(f"problem.dimension < {experiment_count}")

    distance_matrix: np.ndarray = utils.create_distance_matrix(problem)
    paths = []
    times = []
    for _ in range(experiment_count):
        time_start = time.time()
        path = problem_solver.solve(distance_matrix)
        time_end = time.time()
        times.append((time_end - time_start))
        paths.append(path)

    average_time = __process_results(problem, result_title, paths, times)
    return average_time


def run_experiment_iterative_local_search(problem: StandardProblem,
                                          problem_solver: IteratedSearchProblemSolver,
                                          result_title: str = "graph",
                                          experiment_count: int = 100,
                                          max_time: float = None) -> float:
    """ Solves problem using 50 different randomly selected start nodes.

    :param problem: problem which contains graph nodes
    :param problem_solver: specific implementation of ProblemSolver
    :param result_title: title which will be given to result image
    """
    if problem.dimension < experiment_count:
        raise ValueError(f"problem.dimension < {experiment_count}")

    distance_matrix: np.ndarray = utils.create_distance_matrix(problem)
    paths = []
    times = []
    search_invocation_counts = []
    for _ in range(experiment_count):
        time_start = time.time()
        path, search_invocation_count = problem_solver.solve(distance_matrix, max_time=max_time)
        time_end = time.time()
        times.append((time_end - time_start))
        search_invocation_counts.append(search_invocation_count)
        paths.append(path)

    average_time = __process_results(problem, result_title, paths, times, search_invocation_counts)
    return average_time
