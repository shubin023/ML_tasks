from sklearn.datasets import make_blobs, make_moons
import numpy as np
import pandas
import random
from typing import Callable, Union, NoReturn, Optional, Dict, Any, List

# Task 1

def gini(x: np.ndarray) -> float:
    """
    Считает коэффициент Джини для массива меток x.
    """
    _, counts = np.unique(x, return_counts=True)
    probs = counts / counts.sum()
    return np.sum(probs * (1 - probs))

def entropy(x: np.ndarray) -> float:
    """
    Считает коэффициент Джини для массива меток x.
    """
    _, counts = np.unique(x, return_counts=True)
    probs = counts / counts.sum()
    return -np.sum(probs * np.log2(probs))

def gain(left_y: np.ndarray, right_y: np.ndarray, criterion: Callable) -> float:
    """
    Считает информативность разбиения массива меток.

    Parameters
    ----------
    left_y : np.ndarray
        Левая часть разбиения.
    right_y : np.ndarray
        Правая часть разбиения.
    criterion : Callable
        Критерий разбиения.
    """
    all_sides = np.hstack((left_y, right_y))
    l_size = left_y.size
    r_size = right_y.size
    all_size = all_sides.size

    return all_size * criterion(all_sides) - l_size * criterion(left_y) - r_size * criterion(right_y)


# Task 2

class DecisionTreeLeaf:
    """

    Attributes
    ----------
    y : Тип метки (напр., int или str)
        Метка класса, который встречается чаще всего среди элементов листа дерева
    """

    def __init__(self, ys):
        values, counts = np.unique(ys, return_counts=True)
        counts_sum = counts.sum()
        self.probs_dict = {value: count / counts_sum for value, count in zip(values, counts)}
        self.y = values[np.argmax(counts)]

class DecisionTreeNode:
    """

    Attributes
    ----------
    split_dim : int
        Измерение, по которому разбиваем выборку.
    split_value : float
        Значение, по которому разбираем выборку.
    left : Union[DecisionTreeNode, DecisionTreeLeaf]
        Поддерево, отвечающее за случай x[split_dim] < split_value.
    right : Union[DecisionTreeNode, DecisionTreeLeaf]
        Поддерево, отвечающее за случай x[split_dim] >= split_value. 
    """
    def __init__(self, split_dim: int, split_value: float, 
                 left: Union['DecisionTreeNode', DecisionTreeLeaf], 
                 right: Union['DecisionTreeNode', DecisionTreeLeaf]):
        self.split_dim = split_dim
        self.split_value = split_value
        self.left = left
        self.right = right
        
# Task 3

# def tree_constructor(X: np.ndarray, y: np.ndarray, criterion_func, max_depth, cur_depth, min_samples_leaf):
#     samples_num, features_num = X.shape
#     if (max_depth is not None and cur_depth == max_depth) or (samples_num // 2 < min_samples_leaf):
#         return DecisionTreeLeaf(y)
#
#     best_criterion_val, split_feature, split_val = 0, 0, 0
#     left_indices_res, right_indices_res = None, None
#     for feature in range(features_num):
#         split_val = X[0, feature]
#         for value in X[:, feature]:
#             left_indices = X[:, feature] <= value
#             right_indices = X[:, feature] > value
#
#             if left_indices.sum() < min_samples_leaf or right_indices.sum() < min_samples_leaf:
#                 continue
#
#             gain_res = gain(y[left_indices], y[right_indices], criterion_func)
#             if gain_res > best_criterion_val:
#                 best_criterion_val, split_feature, split_val = gain_res, feature, value
#                 left_indices_res, right_indices_res = left_indices, right_indices
#
#     left_side = tree_constructor(X[left_indices_res], y[left_indices_res], criterion_func, max_depth, cur_depth + 1,
#                                  min_samples_leaf)
#     right_side = tree_constructor(X[right_indices_res], y[right_indices_res], criterion_func, max_depth, cur_depth + 1,
#                                  min_samples_leaf)
#
#     return DecisionTreeNode(split_feature, split_val, left_side, right_side)


class DecisionTreeClassifier:
    """
    Attributes
    ----------
    root : Union[DecisionTreeNode, DecisionTreeLeaf]
        Корень дерева.

    (можете добавлять в класс другие аттрибуты).

    """
    def __init__(self, criterion: str = "gini",
                 max_depth: Optional[int] = None,
                 min_samples_leaf: int = 1):
        """
        Parameters
        ----------
        criterion : str
            Задает критерий, который будет использоваться при построении дерева.
            Возможные значения: "gini", "entropy".
        max_depth : Optional[int]
            Ограничение глубины дерева. Если None - глубина не ограничена.
        min_samples_leaf : int
            Минимальное количество элементов в каждом листе дерева.

        """
        self.root = None
        if criterion == "entropy":
            self.criterion_func = entropy
        else:
            self.criterion_func = gini
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.X = None
        self.y = None
        self.classes = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Строит дерево решений по обучающей выборке.

        Parameters
        ----------
        X : np.ndarray
            Обучающая выборка.
        y : np.ndarray
            Вектор меток классов.
        """

        self.X = X
        self.y = y
        self.classes = np.unique(y)
        self.root = self.__tree_constructor(np.array([True] * X.shape[0]), 0)

    def __tree_constructor(self, indices, cur_depth):
        n_samples = indices.sum()
        features_num = self.X.shape[1]
        if (
                (self.max_depth is not None and cur_depth == self.max_depth)
                or (n_samples < 2 * self.min_samples_leaf)
                or (np.unique(self.y[indices]).size == 1)
        ):
            return DecisionTreeLeaf(self.y[indices])

        best_criterion_val = -1.0
        split_feature = None
        split_val = None
        left_indices_res = None
        right_indices_res = None

        for feature in range(features_num):
            unique_values = np.unique(self.X[indices, feature])
            if unique_values.size > 10:
                candidate_values = np.linspace(unique_values[0], unique_values[-1], num=10, endpoint=True)
            else:
                candidate_values = unique_values
            for value in candidate_values:
                left_indices = indices & (self.X[:, feature] <= value)
                right_indices = indices & (self.X[:, feature] > value)

                if left_indices.sum() < self.min_samples_leaf or right_indices.sum() < self.min_samples_leaf:
                    continue

                gain_res = gain(self.y[left_indices], self.y[right_indices], self.criterion_func)
                if gain_res > best_criterion_val:
                    best_criterion_val = gain_res
                    split_feature = feature
                    split_val = value
                    left_indices_res = left_indices
                    right_indices_res = right_indices

        if split_feature is None or best_criterion_val <= 0:
            return DecisionTreeLeaf(self.y[indices])

        left_side = self.__tree_constructor(left_indices_res, cur_depth + 1)
        right_side = self.__tree_constructor(right_indices_res, cur_depth + 1)
        return DecisionTreeNode(split_feature, split_val, left_side, right_side)

    def predict_proba(self, X: np.ndarray) -> List[Dict[Any, float]]:
        """
        Предсказывает вероятность классов для элементов из X.

        Parameters
        ----------
        X : np.ndarray
            Элементы для предсказания.
        
        Return
        ------
        List[Dict[Any, float]]
            Для каждого элемента из X возвращает словарь 
            {метка класса -> вероятность класса}.
        """
        return [self.__prob_dict(object) for object in X]

    def __prob_dict(self, object):
        node = self.root
        while isinstance(node, DecisionTreeNode):
            if object[node.split_dim] <= node.split_value:
                node = node.left
            else:
                node = node.right
        prob_dict = {value: 0.0 for value in self.classes}
        for cls, prob in node.probs_dict.items():
            prob_dict[cls] = prob
        return prob_dict

    
    def predict(self, X : np.ndarray) -> list:
        """
        Предсказывает классы для элементов X.

        Parameters 
        ----------
        X : np.ndarray
            Элементы для предсказания.
        
        Return
        ------
        list
            Вектор предсказанных меток для элементов X.
        """
        proba = self.predict_proba(X)
        return [max(p.keys(), key=lambda k: p[k]) for p in proba]
    
# Task 4
task4_dtc = DecisionTreeClassifier("gini", 5, 30)
