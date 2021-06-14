import os
import shutil
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
import tsplib95
from tsplib95.models import StandardProblem

from common.experiments import run_experiment_iterative_local_search
from common.utils import create_distance_matrix, calculate_path_length
from solvers.local_search import GreedyLocalSearch
from solvers.local_search_multi_improved import ImprovedHybridEvolutionarySolver


def node_similarity(cycle_1: np.ndarray, cycle_2: np.ndarray) -> int:
    sorted_cycle_1 = cycle_1.copy()
    sorted_cycle_1.sort()

    sorted_cycle_2 = cycle_2.copy()
    sorted_cycle_2.sort()

    idx = np.searchsorted(sorted_cycle_1, sorted_cycle_2)
    idx[idx == len(sorted_cycle_1)] = 0
    mask = sorted_cycle_1[idx] == sorted_cycle_2
    return int(np.sum(np.bincount(idx[mask])))


def edge_similarity(cycle_1: np.ndarray, cycle_2: np.ndarray) -> int:
    similiarity = 0
    for i in np.arange(start=0, stop=cycle_1.shape[0] - 1):
        for j in np.arange(start=0, stop=cycle_2.shape[0] - 1):
            if cycle_1[i:i + 1] == cycle_2[j:j + 1]:
                similiarity += 1
    return similiarity


def global_convexity_tests(problem: StandardProblem,
                           number_of_solutions: int = 1000,
                           similarity_function=node_similarity,
                           title: str = ""):
    distance_matrix = create_distance_matrix(problem)

    problem_solver = GreedyLocalSearch(use_node_swap=True)

    pool = Pool(processes=os.cpu_count())
    pool_results = []
    solutions = []
    for i_ in range(number_of_solutions):
        if len(pool_results) == os.cpu_count():
            for pool_res in pool_results:
                solution = pool_res.get()
                solutions.append(np.array(solution))
            pool_results.clear()

        res = pool.apply_async(problem_solver.solve, (distance_matrix,))
        pool_results.append(res)

    for pool_res in pool_results:
        solution = pool_res.get()
        solutions.append(np.array(solution))
    pool_results.clear()
    pool.close()

    solution_cost = np.array([calculate_path_length(distance_matrix, list(cycle)) for cycle in solutions])
    best_cost_index = np.argmin(solution_cost)
    best_solution = solutions[best_cost_index]

    similarity = \
        np.array([similarity_function(cycle, best_solution) for cycle in solutions])

    average_other_similarity = np.zeros(shape=similarity.shape)
    best_solution_similarity = np.zeros(shape=similarity.shape)
    for i in np.arange(start=0, stop=similarity.shape[0]):
        mask = np.ones(similarity.shape, bool)
        mask[i] = False
        average_other_similarity[i] = np.average(similarity[mask])
        best_solution_similarity[i] = similarity_function(solutions[i], best_solution)

    correlation = np.corrcoef(solution_cost, average_other_similarity)[0][1]
    print(f"Correlation parameter ({title}) : {correlation}")

    indices = np.argsort(solution_cost)

    plt.scatter(solution_cost[indices], average_other_similarity[indices], label="Average other similarity")
    plt.scatter(solution_cost[indices], best_solution_similarity[indices], label="Best solution similarity")
    plt.legend()
    plt.title(title)
    plt.savefig(f"./graphs/{title}.pdf")
    plt.show()


_COUNT: int = 10


def main():
    problem_a: StandardProblem = tsplib95.load('data/kroa200.tsp')
    problem_b: StandardProblem = tsplib95.load('data/krob200.tsp')

    shutil.rmtree("./graphs/", ignore_errors=True)
    os.makedirs('graphs/')

    global_convexity_tests(problem_a, title="problem_a_node")
    global_convexity_tests(problem_a, similarity_function=edge_similarity, title="problem_a_edge")
    global_convexity_tests(problem_b, title="problem_b_node")
    global_convexity_tests(problem_b, similarity_function=edge_similarity, title="problem_b_edge")

    run_experiment_iterative_local_search(problem_a,
                                          ImprovedHybridEvolutionarySolver(),
                                          result_title="kroa200_ihe",
                                          experiment_count=_COUNT,
                                          max_time=20.777)

    run_experiment_iterative_local_search(problem_b,
                                          ImprovedHybridEvolutionarySolver(),
                                          result_title="krob200_ihe",
                                          experiment_count=_COUNT,
                                          max_time=20.205)


if __name__ == '__main__':
    main()
