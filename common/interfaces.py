from abc import ABC, abstractmethod
from typing import List

import numpy as np


class ProblemSolver(ABC):

    @abstractmethod
    def solve(self, distance_matrix: np.ndarray, start_node: int = 1) -> List[int]:
        """ Creates path for given graph.

        :param distance_matrix: distance_matrix of given graph
        :param start_node: index of path start node
        :return: Path generated by given implementation of problem solver
        """
        pass


class SearchProblemSolver(ABC):

    @abstractmethod
    def solve(self, distance_matrix: np.ndarray) -> List[int]:
        """ Creates path for given graph.

        :param distance_matrix: distance_matrix of given graph
        :return: Path generated by given implementation of problem solver
        """
        pass
