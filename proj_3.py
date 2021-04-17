import shutil

import tsplib95
from tsplib95.models import StandardProblem

from common.experiments import run_experiment_local_search
from solvers.local_search_improved import CandidateSteepSearch, ScoreSteepSearch

_EXPERIMENT_COUNT: int = 100


def main():
    problem_a: StandardProblem = tsplib95.load('data/kroa200.tsp')
    problem_b: StandardProblem = tsplib95.load('data/krob200.tsp')

    shutil.rmtree("./graphs/", ignore_errors=True)

    run_experiment_local_search(problem_a, CandidateSteepSearch(), "kroa200_css", _EXPERIMENT_COUNT)
    run_experiment_local_search(problem_b, CandidateSteepSearch(), "krob200_css", _EXPERIMENT_COUNT)
    run_experiment_local_search(problem_a, ScoreSteepSearch(), "kroa200_sss", _EXPERIMENT_COUNT)
    run_experiment_local_search(problem_b, ScoreSteepSearch(), "krob200_sss", _EXPERIMENT_COUNT)


if __name__ == '__main__':
    main()
