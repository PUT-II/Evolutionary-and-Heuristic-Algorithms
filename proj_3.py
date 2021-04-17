import shutil

import tsplib95
from tsplib95.models import StandardProblem

from common.experiments import run_experiment_local_search
from solvers.local_search_improved import EdgeSwapSteepSearch

_EXPERIMENT_COUNT: int = 100


def main():
    problem_a: StandardProblem = tsplib95.load('data/kroa200.tsp')
    problem_b: StandardProblem = tsplib95.load('data/krob200.tsp')

    shutil.rmtree("./graphs/", ignore_errors=True)

    run_experiment_local_search(problem_a, EdgeSwapSteepSearch(), "kroa200_ss")
    run_experiment_local_search(problem_b, EdgeSwapSteepSearch(), "krob200_ss")


if __name__ == '__main__':
    main()
