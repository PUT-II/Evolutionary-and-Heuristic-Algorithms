import shutil

import tsplib95
from tsplib95.models import StandardProblem

from common.experiments import run_experiment_local_search
from solvers.local_search_improved import CandidateSteepSearch
from solvers.local_search_multi import MultipleStartLocalSearch, IteratedLocalSearch1

_EXPERIMENT_COUNT: int = 10


def main():
    problem_a: StandardProblem = tsplib95.load('data/kroa200.tsp')
    problem_b: StandardProblem = tsplib95.load('data/krob200.tsp')

    shutil.rmtree("./graphs/", ignore_errors=True)

    run_experiment_local_search(problem_a, CandidateSteepSearch(), "kroa200_css", _EXPERIMENT_COUNT)
    run_experiment_local_search(problem_b, CandidateSteepSearch(), "krob200_css", _EXPERIMENT_COUNT)
    avg_time_kroa = run_experiment_local_search(problem_a, MultipleStartLocalSearch(), "kroa200_msls",
                                                _EXPERIMENT_COUNT)
    avg_time_krob = run_experiment_local_search(problem_b, MultipleStartLocalSearch(), "krob200_msls",
                                                _EXPERIMENT_COUNT)
    run_experiment_local_search(problem_a, IteratedLocalSearch1(), "kroa200_ils1", _EXPERIMENT_COUNT, avg_time_kroa)
    run_experiment_local_search(problem_b, IteratedLocalSearch1(), "krob200_ils1", _EXPERIMENT_COUNT, avg_time_krob)


if __name__ == '__main__':
    main()
