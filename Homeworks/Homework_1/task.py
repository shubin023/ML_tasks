import numpy as np
import random
import copy
import pandas
from typing import NoReturn, Tuple, List

import pandas as pd


# Task 1

def read_cancer_dataset(path_to_csv: str) -> Tuple[np.array, np.array]:
    """
     
    Parameters
    ----------
    path_to_csv : str
        Путь к cancer датасету.

    Returns
    -------
    X : np.array
        Матрица признаков опухолей.
    y : np.array
        Вектор бинарных меток, 1 соответствует доброкачественной опухоли (M), 
        0 --- злокачественной (B).

    """
    data = pd.read_csv(path_to_csv).sample(frac=1, random_state=42).reset_index(drop=True)

    X = data.drop(columns='label').to_numpy()
    y = data.label.replace(['B', 'M'], [0, 1]).to_numpy()

    return X, y


def read_spam_dataset(path_to_csv: str) -> Tuple[np.array, np.array]:
    """
     
    Parameters
    ----------
    path_to_csv : str
        Путь к spam датасету.

    Returns
    -------
    X : np.array
        Матрица признаков сообщений.
    y : np.array
        Вектор бинарных меток, 
        1 если сообщение содержит спам, 0 если не содержит.
    
    """

    data = pd.read_csv(path_to_csv).sample(frac=1, random_state=42).reset_index(drop=True)

    X = data.drop(columns='label').to_numpy()
    y = data.label.to_numpy()

    return X, y

    
# Task 2

def train_test_split(X: np.array, y: np.array, ratio: float) -> Tuple[np.array, np.array, np.array, np.array]:
    """

    Parameters
    ----------
    X : np.array
        Матрица признаков.
    y : np.array
        Вектор меток.
    ratio : float
        Коэффициент разделения.

    Returns
    -------
    X_train : np.array
        Матрица признаков для train выборки.
    y_train : np.array
        Вектор меток для train выборки.
    X_test : np.array
        Матрица признаков для test выборки.
    y_test : np.array
        Вектор меток для test выборки.

    """

    train_size = round(y.size * ratio)

    X_train = X[:train_size, :]
    y_train = y[:train_size]

    X_test = X[train_size:, :]
    y_test = y[train_size:]

    return X_train, y_train, X_test, y_test


# Task 3

def get_precision_recall_accuracy(y_pred: np.array, y_true: np.array) -> Tuple[np.array, np.array, float]:
    """

    Parameters
    ----------
    y_pred : np.array
        Вектор классов, предсказанных моделью.
    y_true : np.array
        Вектор истинных классов.

    Returns
    -------
    precision : np.array
        Вектор с precision для каждого класса.
    recall : np.array
        Вектор с recall для каждого класса.
    accuracy : float
        Значение метрики accuracy (одно для всех классов).

    """

    classes = np.unique(y_true)
    class_num = classes.size
    sample_size = y_true.size

    TP_arr = np.zeros(class_num)
    FP_arr = np.zeros(class_num)
    FN_arr = np.zeros(class_num)

    for idx, cls in enumerate(classes):
        TP_arr[idx] = np.sum((y_pred == cls) & (y_true == cls))
        FP_arr[idx] = np.sum((y_pred == cls) & (y_true != cls))
        FN_arr[idx] = np.sum((y_pred != cls) & (y_true == cls))

    precision = TP_arr / (TP_arr + FP_arr + 1e-10)
    recall = TP_arr / (TP_arr + FN_arr + 1e-10)

    accuracy = np.sum(y_pred == y_true) / sample_size

    return precision, recall, accuracy


# Task 4

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

        def _search(node, point, best_neighbors):
            if isinstance(node, Leaf):
                for idx in node.indices:
                    dist = np.linalg.norm(point - self.data[idx])
                    if len(best_neighbors) < k:
                        best_neighbors.append((dist, idx))
                        best_neighbors.sort()
                    elif dist < best_neighbors[-1][0]:
                        best_neighbors[-1] = (dist, idx)
                        best_neighbors.sort()
                return

            axis = node.split_axis
            split_value = node.split_value

            if point[axis] <= split_value:
                next_branch, opposite_branch = node.left, node.right
            else:
                next_branch, opposite_branch = node.right, node.left

            _search(next_branch, point, best_neighbors)

            if len(best_neighbors) < k or abs(point[axis] - split_value) < best_neighbors[-1][0]:
                _search(opposite_branch, point, best_neighbors)

        results = []
        for query_point in X:
            best_neighbors = []
            _search(self.tree, query_point, best_neighbors)
            results.append([idx for _, idx in best_neighbors])
        return results

# Task 5

class KNearest:
    def __init__(self, n_neighbors: int = 5, leaf_size: int = 30):
        """

        Parameters
        ----------
        n_neighbors : int
            Число соседей, по которым предсказывается класс.
        leaf_size : int
            Минимальный размер листа в KD-дереве.

        """

        self.n_neighbors = n_neighbors
        self.leaf_size = leaf_size
        self.labels = None
        self.classes = None
        self.classifier = None


    def fit(self, X: np.array, y: np.array) -> NoReturn:
        """

        Parameters
        ----------
        X : np.array
            Набор точек, по которым строится классификатор.
        y : np.array
            Метки точек, по которым строится классификатор.

        """

        self.classifier = KDTree(X, self.leaf_size)
        self.labels = y
        self.classes = np.unique(y)


    def predict_proba(self, X: np.array) -> List[np.array]:
        """

        Parameters
        ----------
        X : np.array
            Набор точек, для которых нужно определить класс.

        Returns
        -------
        list[np.array]
            Список np.array (длина каждого np.array равна числу классов):
            вероятности классов для каждой точки X.


        """

        result = []
        close_neighbors = self.classifier.query(X, self.n_neighbors)
        for object_neighbors in close_neighbors:
            prob_arr = np.zeros(self.classes.size)
            for idx, cls in enumerate(self.classes):
                prob_arr[idx] = np.sum(self.labels[object_neighbors] == cls) / self.labels.size
            result.append(prob_arr)

        return result



    def predict(self, X: np.array) -> np.array:
        """

        Parameters
        ----------
        X : np.array
            Набор точек, для которых нужно определить класс.

        Returns
        -------
        np.array
            Вектор предсказанных классов.


        """
        
        return np.argmax(self.predict_proba(X), axis=1)
