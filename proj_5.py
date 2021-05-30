import shutil

import tsplib95
from tsplib95.models import StandardProblem

from common.experiments import run_experiment_iterative_local_search
from solvers.hybrid_evolutionary import HybridEvolutionarySolver

_EXPERIMENT_COUNT: int = 10


def main():
    problem_a: StandardProblem = tsplib95.load('data/kroa200.tsp')
    problem_b: StandardProblem = tsplib95.load('data/krob200.tsp')

    shutil.rmtree("./graphs/", ignore_errors=True)

    run_experiment_iterative_local_search(problem_a, HybridEvolutionarySolver(), "kroa200_he", _EXPERIMENT_COUNT,
                                          20.777)

    run_experiment_iterative_local_search(problem_b, HybridEvolutionarySolver(), "krob200_he", _EXPERIMENT_COUNT,
                                          20.205)


if __name__ == '__main__':
    main()
