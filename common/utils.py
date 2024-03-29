from typing import List

import numpy as np
from tsplib95.models import StandardProblem

_SHOW_NODE_LABELS: bool = True


def calculate_distance(point_1: List[int], point_2: List[int]) -> int:
    """ Calculates distance between given points.

    :param point_1: point coordinates (x, y)
    :param point_2: point coordinates (x, y)
    :return: distance between given points
    """
    pow_x: int = (point_1[0] - point_2[0]) ** 2
    pow_y: int = (point_1[1] - point_2[1]) ** 2

    return round((pow_x + pow_y) ** 0.5)


def create_distance_matrix(problem: StandardProblem) -> np.ndarray:
    """ Creates distance matrix for given graph.

    :param problem: problem which contains graph nodes
    :return: distance matrix
    """
    matrix = np.zeros(shape=(problem.dimension, problem.dimension), dtype=np.int32)

    node_dict = dict(problem.node_coords)

    for node_index_1 in range(problem.dimension - 1):
        node_1 = node_dict[node_index_1 + 1]
        for node_index_2 in range(node_index_1 + 1, problem.dimension):
            node_2 = node_dict[node_index_2 + 1]
            distance = calculate_distance(node_1, node_2)
            matrix[node_index_1, node_index_2] = distance
            matrix[node_index_2, node_index_1] = distance

    return matrix


def draw_graph(problem: StandardProblem, path: List[int], result_title: str, path_length: int):
    import os
    import matplotlib.pyplot as plt
    import networkx as nx
    global _SHOW_NODE_LABELS

    start_node = path[0]
    graph = nx.Graph()

    node_coords_dict = dict(problem.node_coords)
    node_colors = []
    for node_index in node_coords_dict:
        graph.add_node(node_index, pos=tuple(node_coords_dict[node_index]))
        node_colors.append('red' if node_index == start_node else 'blue')

    for i in range(len(path) - 1):
        graph.add_edge(path[i], path[i + 1])

    pos = nx.get_node_attributes(graph, 'pos')

    _, ax = plt.subplots()
    nx.draw_networkx_edges(graph, pos, edge_color='red')
    nx.draw_networkx_nodes(graph, pos, node_size=1, node_color=node_colors, ax=ax)
    if _SHOW_NODE_LABELS:
        nx.draw_networkx_labels(graph, pos, font_size=5)

    if not os.path.exists('graphs/'):
        os.makedirs('graphs/')

    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    plt.suptitle(result_title)
    plt.title(f"Length : {path_length}")
    plt.savefig(f"./graphs/{result_title}.pdf")
    plt.show()
    plt.clf()


def calculate_path_length(distance_matrix: np.ndarray, path: list) -> int:
    total_length = 0
    for i in range(len(path) - 1):
        total_length += distance_matrix[path[i], path[i + 1]]

    return total_length


def validate_shape(shape: tuple) -> tuple:
    if len(shape) != 2 or shape[0] != shape[1]:
        raise ValueError('Wrong distance matrix shape: len(shape) != 2 or shape[0] != shape[1]')
    return shape
