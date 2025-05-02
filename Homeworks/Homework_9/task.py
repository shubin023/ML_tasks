from sklearn.model_selection import train_test_split
import numpy as np
from typing import Callable, Union, NoReturn, Optional, Dict, Any, List
import pandas
import random
import copy
from catboost import CatBoostClassifier

# Task 0

def gini(x: np.ndarray) -> float:
    """
    Считает коэффициент Джини для массива меток x.
    """
    _, counts = np.unique(x, return_counts=True)
    probs = counts / counts.sum()
    return np.sum(probs * (1 - probs))
    
def entropy(x: np.ndarray) -> float:
    """
    Считает энтропию для массива меток x.
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
    return (all_size * criterion(all_sides)
            - l_size * criterion(left_y)
            - r_size * criterion(right_y))


class DecisionTreeLeaf:
    """

    Attributes
    ----------
    y : Тип метки (напр., int или str)
        Метка класса, который встречается чаще всего среди элементов листа дерева
    """

    def __init__(self, ys):
        values, counts = np.unique(ys, return_counts=True)
        self.probs_dict = {
            v: c / counts.sum() for v, c in zip(values, counts)
        }
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

    def __init__(self,
                 split_dim: int,
                 left: Union['DecisionTreeNode', DecisionTreeLeaf],
                 right: Union['DecisionTreeNode', DecisionTreeLeaf]):
        self.split_dim = split_dim
        self.left = left
        self.right = right


# Task 1

class DecisionTree:
    """
    Attributes
    ----------
    root : Union[DecisionTreeNode, DecisionTreeLeaf]
        Корень дерева.

    (можете добавлять в класс другие аттрибуты).

    """

    def __init__(self,
                 X: np.ndarray,
                 y: np.ndarray,
                 criterion: str = "gini",
                 max_depth: Optional[int] = None,
                 min_samples_leaf: int = 1,
                 max_features: Union[int, str] = "auto"):

        """
        Parameters
        ----------
        X, y - обучающая выборка и соответствующие ей метки классов.
            Из нее нужно получить выборку для построения дерева при помощи bagging.
            Out-of-bag выборку нужно запомнить, она понадобится потом.
        criterion : str
            Задает критерий, который будет использоваться при построении дерева.
            Возможные значения: "gini", "entropy".
        max_depth : Optional[int]
            Ограничение глубины дерева. Если None - глубина не ограничена.
        min_samples_leaf : int
            Минимальное количество элементов в каждом листе дерева.
        max_features - количество признаков, которые могут использоваться в узле.
            Если "auto" - равно sqrt(X.shape[1])

        """
        if criterion == "entropy":
            self.criterion_func = entropy
        else:
            self.criterion_func = gini

        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf

        if max_features == "auto":
            self.max_features = int(np.sqrt(X.shape[1]))
        else:
            self.max_features = max_features

        (self.X_bootstrap,
         self.y_bootstrap,
         self.X_oob,
         self.y_oob) = self.__bagging_with_oob(X, y)

        self.features_num = self.X_bootstrap.shape[1]
        self.classes = np.unique(self.y_bootstrap)

        all_indices = np.arange(self.X_bootstrap.shape[0])
        self.root = self.__tree_constructor(all_indices, 0)

    def get_features_num(self):
        return self.features_num

    def __bagging_with_oob(self, X, y):
        n_samples = X.shape[0]
        bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)

        X_bootstrap = X[bootstrap_indices]
        y_bootstrap = y[bootstrap_indices]

        mask = np.ones(n_samples, dtype=bool)
        mask[bootstrap_indices] = False

        X_oob = X[mask]
        y_oob = y[mask]

        return X_bootstrap, y_bootstrap, X_oob, y_oob

    def __tree_constructor(self,
                           indices: np.ndarray,
                           cur_depth: int):

        y_subset = self.y_bootstrap[indices]
        n_samples = indices.sum()

        if self.max_depth is not None and cur_depth >= self.max_depth:
            return DecisionTreeLeaf(y_subset)
        if n_samples < 2 * self.min_samples_leaf:
            return DecisionTreeLeaf(y_subset)
        if np.unique(y_subset).size == 1:
            return DecisionTreeLeaf(y_subset)

        best_criterion_val = -1.0
        split_feature = None
        left_indices_res = None
        right_indices_res = None

        features = np.random.choice(self.features_num, self.max_features, replace=False)

        for feature in features:

            x_col = self.X_bootstrap[indices, feature]

            left_mask = (x_col == 0)
            right_mask = (x_col == 1)

            if left_mask.sum() < self.min_samples_leaf or right_mask.sum() < self.min_samples_leaf:
                continue

            left_indices = indices[left_mask]
            right_indices = indices[right_mask]

            y_left = self.y_bootstrap[left_indices]
            y_right = self.y_bootstrap[right_indices]
            gain_res = gain(y_left, y_right, self.criterion_func)

            if gain_res > best_criterion_val:
                best_criterion_val = gain_res
                split_feature = feature
                left_indices_res = left_indices
                right_indices_res = right_indices

        if split_feature is None or best_criterion_val <= 0:
            return DecisionTreeLeaf(y_subset)

        left_side = self.__tree_constructor(left_indices_res, cur_depth + 1)
        right_side = self.__tree_constructor(right_indices_res, cur_depth + 1)
        return DecisionTreeNode(split_feature, left_side, right_side)

    def predict_proba(self, X: np.ndarray) -> List[Dict[Any, float]]:
        probabilities = []
        for x in X:
            node = self.root
            while isinstance(node, DecisionTreeNode):
                if x[node.split_dim] == 0:
                    node = node.left
                else:
                    node = node.right

            prob_dict = {cls: 0.0 for cls in self.classes}
            for cls, prob in node.probs_dict.items():
                prob_dict[cls] = prob
            probabilities.append(prob_dict)
        return probabilities

    def predict(self, X: np.ndarray) -> list:
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
        preds = []
        for x in X:
            node = self.root
            while isinstance(node, DecisionTreeNode):
                if x[node.split_dim] == 0:
                    node = node.left
                else:
                    node = node.right
            preds.append(node.y)

        return np.array(preds)
    
# Task 2

class RandomForestClassifier:
    def __init__(self,
                 criterion="gini",
                 max_depth=None,
                 min_samples_leaf=1,
                 max_features="auto",
                 n_estimators=10):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.n_estimators = n_estimators
        self.trees_list = []
    
    def fit(self, X, y):
        for _ in range(self.n_estimators):
            tree = DecisionTree(X, y, self.criterion,
                                self.max_depth,
                                self.min_samples_leaf,
                                self.max_features)
            self.trees_list.append(tree)

    def predict(self, X):
        all_preds = np.array([tree.predict(X) for tree in self.trees_list])
        final_preds = []
        for i in range(all_preds.shape[1]):
            values, counts = np.unique(all_preds[:, i], return_counts=True)
            final_preds.append(values[np.argmax(counts)])
        return np.array(final_preds)

    def get_feature_importance(self):
        n_features = self.trees_list[0].get_features_num()
        importance = np.zeros(n_features, dtype=float)

        n_trees_used = 0

        for tree in self.trees_list:
            X_oob = tree.X_oob
            y_oob = tree.y_oob

            if X_oob.shape[0] == 0:
                continue

            y_pred = tree.predict(X_oob)
            err_oob = (y_pred != y_oob).mean()

            for j in range(n_features):
                X_oob_perm = X_oob.copy()
                np.random.shuffle(X_oob_perm[:, j])

                y_pred_perm = tree.predict(X_oob_perm)
                err_oob_j = (y_pred_perm != y_oob).mean()

                importance[j] += (err_oob_j - err_oob)

            n_trees_used += 1

        if n_trees_used > 0:
            importance /= n_trees_used

        return importance
    
# Task 3

def feature_importance(rfc):
    return rfc.get_feature_importance()


# Task 4

rfc_age = RandomForestClassifier(criterion="gini",
                                 max_depth=20,
                                 min_samples_leaf=40,
                                 max_features="auto",
                                 n_estimators=10)

rfc_gender = RandomForestClassifier(criterion="gini",
                                    max_depth=20,
                                    min_samples_leaf=40,
                                    max_features="auto",
                                    n_estimators=10)

# Task 5
# Здесь нужно загрузить уже обученную модели
# https://catboost.ai/en/docs/concepts/python-reference_catboost_save_model
# https://catboost.ai/en/docs/concepts/python-reference_catboost_load_model
catboost_rfc_age = CatBoostClassifier()
catboost_rfc_age.load_model(__file__[:-7] + '/' + 'cat_age.cbm')

catboost_rfc_gender = CatBoostClassifier()
catboost_rfc_gender.load_model(__file__[:-7] + '/' + 'cat_sex.cbm')