import shutil

import tsplib95
from tsplib95.models import StandardProblem

from common.experiments import run_experiment_local_search
from solvers.local_search import RandomSearch, EdgeSwapSteepSearch, NodeSwapSteepSearch, GreedyLocalSearch

_EXPERIMENT_COUNT: int = 100


def main():
    problem_a: StandardProblem = tsplib95.load('data/kroa100.tsp')
    problem_b: StandardProblem = tsplib95.load('data/krob100.tsp')

    shutil.rmtree("./graphs/", ignore_errors=True)

    average_time = []
    average_time.append(
        run_experiment_local_search(problem_a, NodeSwapSteepSearch(), "kroa100_nsss", _EXPERIMENT_COUNT))

    average_time.append(
        run_experiment_local_search(problem_a, EdgeSwapSteepSearch(), "kroa100_esss", _EXPERIMENT_COUNT))

    average_time.append(run_experiment_local_search(problem_a, GreedyLocalSearch(use_node_swap=True), "kroa100_gls_ns",
                                                    _EXPERIMENT_COUNT))

    average_time.append(run_experiment_local_search(problem_a, GreedyLocalSearch(use_edge_swap=True), "kroa100_gls_es",
                                                    _EXPERIMENT_COUNT))

    average_time.append(
        run_experiment_local_search(problem_b, NodeSwapSteepSearch(), "krob100_nsss", _EXPERIMENT_COUNT))

    average_time.append(
        run_experiment_local_search(problem_b, EdgeSwapSteepSearch(), "krob100_esss", _EXPERIMENT_COUNT))

    average_time.append(run_experiment_local_search(problem_b, GreedyLocalSearch(use_node_swap=True), "krob100_gls_ns",
                                                    _EXPERIMENT_COUNT))

    average_time.append(run_experiment_local_search(problem_b, GreedyLocalSearch(use_edge_swap=True), "krob100_gls_es",
                                                    _EXPERIMENT_COUNT))

    max_avg_time = max(average_time)
    print(f"Max average time : {round(max_avg_time * 1000.0)}")
    print()
    run_experiment_local_search(problem_a, RandomSearch(max_avg_time), "kroa100_rs", _EXPERIMENT_COUNT)
    run_experiment_local_search(problem_b, RandomSearch(max_avg_time), "krob100_rs", _EXPERIMENT_COUNT)


if __name__ == '__main__':
    main()
