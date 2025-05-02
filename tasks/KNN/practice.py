import numpy as np
import random
import copy
import pandas
from typing import NoReturn, Tuple, List

import pandas as pd

class Leaf:
    def __init__(self, indices):
        self.indices = indices

class Node:
    def __init__(self, split_axis: int, split_value: float, left, right):
        self.split_axis = split_axis
        self.split_value = split_value
        self.left = left
        self.right = right

class KDTree:
    def __init__(self, X: np.array, leaf_size: int = 40):
        """

        Parameters
        ----------
        X : np.array
            Набор точек, по которому строится дерево.
        leaf_size : int
            Минимальный размер листа
            (то есть, пока возможно, пространство разбивается на области,
            в которых не меньше leaf_size точек).

        Returns
        -------

        """
        self.data = X
        self.leaf_size = leaf_size
        self.tree = self._build_tree(np.arange(len(X)))

    def _build_tree(self, indices, depth: int = 0):
        n_points = indices.size
        if n_points <= self.leaf_size:
            return Leaf(indices)

        axis = depth % self.data.shape[1]
        median_idx = n_points // 2
        partitioned_indices = indices[np.argpartition(self.data[indices, axis], median_idx)]

        return Node(
            split_axis=axis,
            split_value=self.data[partitioned_indices[median_idx], axis],
            left=self._build_tree(partitioned_indices[:median_idx], depth + 1),
            right=self._build_tree(partitioned_indices[median_idx:], depth + 1)
        )

    def query(self, X: np.array, k: int = 1) -> List[List]:
        """

        Parameters
        ----------
        X : np.array
            Набор точек, для которых нужно найти ближайших соседей.
        k : int
            Число ближайших соседей.

        Returns
        -------
        list[list]
            Список списков (длина каждого списка k):
            индексы k ближайших соседей для всех точек из X.

        """

        def _search(node, point):
            nonlocal best_neighbors
            nonlocal point_leaf_visit_flg

            if isinstance(node, Leaf):
                if not point_leaf_visit_flg:
                    point_leaf_visit_flg = True
                    for idx in node.indices:
                        dist = np.linalg.norm(point - self.data[idx])
                        best_neighbors.append((dist, idx))
                    best_neighbors.sort()
                    best_neighbors = best_neighbors[:k]
                else:
                    max_dist = max([d[0] for d in best_neighbors])
                    for idx in node.indices:
                        dist = np.linalg.norm(point - self.data[idx])
                        if dist < max_dist or len(best_neighbors) < k:
                            best_neighbors.append((dist, idx))
                return

            axis = node.split_axis
            split_value = node.split_value

            if point[axis] <= split_value:
                next_branch = node.left
                opposite_branch = node.right
            else:
                next_branch = node.right
                opposite_branch = node.left

            _search(next_branch, point)

            if len(best_neighbors) < k or abs(point[axis] - split_value) < max([d[0] for d in best_neighbors]):
                _search(opposite_branch, point)

        results = []
        for query_point in X:
            best_neighbors = []
            point_leaf_visit_flg = False

            _search(self.tree, query_point)

            best_neighbors = sorted(best_neighbors)[:k]
            results.append([idx for _, idx in best_neighbors])

        return results

def true_closest(X_train, X_test, k):
    result = []
    for x0 in X_test:
        bests = list(sorted([(i, np.linalg.norm(x - x0)) for i, x in enumerate(X_train)], key=lambda x: x[1]))
        bests = [i for i, d in bests]
        result.append(bests[:min(k, len(bests))])
    return result

X_train = np.random.randn(100, 3)
X_test = np.random.randn(10, 3)
tree = KDTree(X_train, leaf_size=2)
predicted = tree.query(X_test, k=4)
true = true_closest(X_train, X_test, k=4)

