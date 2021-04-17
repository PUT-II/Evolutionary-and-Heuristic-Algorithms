import shutil

import tsplib95
from tsplib95.models import StandardProblem

from common.experiments import run_experiment_constructive
from solvers.problem_solvers import GreedyCycleProblemSolver, NearestNeighbourProblemSolver, RegretCycleProblemSolver

_EXPERIMENT_COUNT: int = 50


def main():
    problem_a: StandardProblem = tsplib95.load('data/kroa100.tsp')
    problem_b: StandardProblem = tsplib95.load('data/krob100.tsp')

    shutil.rmtree("graphs/", ignore_errors=True)

    run_experiment_constructive(problem_a, NearestNeighbourProblemSolver(), "kroa100_nn", _EXPERIMENT_COUNT)
    run_experiment_constructive(problem_b, NearestNeighbourProblemSolver(), "krob100_nn", _EXPERIMENT_COUNT)
    run_experiment_constructive(problem_a, GreedyCycleProblemSolver(), "kroa100_gc", _EXPERIMENT_COUNT)
    run_experiment_constructive(problem_b, GreedyCycleProblemSolver(), "krob100_gc", _EXPERIMENT_COUNT)
    run_experiment_constructive(problem_a, RegretCycleProblemSolver(), "kroa100_rc", _EXPERIMENT_COUNT)
    run_experiment_constructive(problem_b, RegretCycleProblemSolver(), "krob100_rc", _EXPERIMENT_COUNT)


if __name__ == '__main__':
    main()
