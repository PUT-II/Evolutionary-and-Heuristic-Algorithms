from abc import ABC, abstractmethod
from typing import List

from tsplib95.models import StandardProblem


class ProblemSolver(ABC):
    @staticmethod
    def calculate_distance(point_1: List[int], point_2: List[int]) -> float:
        pow_x: int = (point_1[0] - point_2[0]) ** 2
        pow_y: int = (point_1[1] - point_2[1]) ** 2

        return (pow_x + pow_y) ** 0.5

    @abstractmethod
    def solve(self, problem: StandardProblem, start_node: int = 1) -> List[int]:
        pass
