import numpy as np
import pandas as pd
from typing import TypeVar, Dict, List

Array = TypeVar('Array')


def find_adj_matrix(matrix: Array, threshold: float) -> Array:
    """
    Return an adjacency matrix to given a threshold.

    :param matrix: The correlation matrix
    : param threshold: The threshold to determine a path between two vertices
    """
    adj_matrix = list()

    for row in range(len(matrix)):
        adj_row = list()
        for column in range(len(matrix[0])):
            if matrix[row][column] > threshold:
                adj_row.append(1)
            else:
                adj_row.append(0)

        adj_matrix.append(adj_row)

    return np.array(adj_matrix)


def cliques_in_group(reference: Dict, threshold: float) -> Dict:
    """
    Define cliques to each substitution group

    :param reference: The dictionary of all series evaluated
    :param threshold: The threshold to determine a path between two vertices
    """
    from collections import defaultdict
    import networkx as nx

    reference = pd.DataFrame(reference)

    adj_mat = find_adj_matrix(reference.corr().values, threshold)
    G = nx.from_numpy_array(adj_mat)
    cliques = list(nx.find_cliques(G))

    _cliques_on_item = defaultdict(list)

    lookup_dict = {key: val for key, val in enumerate(reference.keys())}

    for idx, item in enumerate(reference.keys()):
        for clique in cliques:
            if idx in clique:
                for number in clique:
                    temp = lookup_dict[number]
                    _cliques_on_item[item].append(temp)

    # only unique values
    for group in _cliques_on_item.keys():
        temp = _cliques_on_item[group]
        temp = sorted(list(set(temp)))
        _cliques_on_item[group] = temp

    return _cliques_on_item
