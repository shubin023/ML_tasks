from sklearn.neighbors import KDTree
import numpy as np
import heapq
import random
import copy
from collections import deque
from typing import NoReturn

# Task 1

class KMeans:
    def __init__(self, n_clusters: int, init: str = "random", 
                 max_iter: int = 300):
        """
        
        Parameters
        ----------
        n_clusters : int
            Число итоговых кластеров при кластеризации.
        init : str
            Способ инициализации кластеров. Один из трех вариантов:
            1. random --- центроиды кластеров являются случайными точками,
            2. sample --- центроиды кластеров выбираются случайно из X,
            3. k-means++ --- центроиды кластеров инициализируются 
                при помощи метода K-means++.
        max_iter : int
            Максимальное число итераций для kmeans.
        
        """
        
        self.k = n_clusters
        self.init_type = init
        self.max_iter = max_iter
        self.X = None
        self.centers = None
        self.clusters = None

    def re_init(self, init_type='random', dist_arr=None):
        if init_type == 'random':
            return np.random.uniform(self.X.min(axis=0), self.X.max(axis=0))
        elif init_type == 'sample':
            return self.X[np.random.choice(self.X.shape[0], replace=False)]
        elif init_type == 'k-means++':
            return self.X[np.random.choice(self.X.shape[0], p=dist_arr / dist_arr.sum())]

    def fit(self, X: np.array, y = None) -> NoReturn:
        """
        Ищет и запоминает в self.centroids центроиды кластеров для X.
        
        Parameters
        ----------
        X : np.array
            Набор данных, который необходимо кластеризовать.
        y : Ignored
            Не используемый параметр, аналогично sklearn
            (в sklearn считается, что все функции fit обязаны принимать 
            параметры X и y, даже если y не используется).
        
        """
        self.X = X

        if self.init_type == 'random':
            self.centers = np.random.uniform(self.X.min(axis=0), self.X.min(axis=0), size=(self.k, self.X.shape[1]))

        elif self.init_type == 'sample':
            unique_ids = np.random.choice(self.X.shape[0], self.k, replace=False)
            self.centers = self.X[unique_ids]

        elif self.init_type == 'k-means++':
            centers_ind = []
            first_ind = np.random.randint(0, self.X.shape[0] - 1)
            centers_ind.append(first_ind)
            dist_arr_sqr = np.linalg.norm(self.X[first_ind] - self.X, axis=1) ** 2
            for _ in range(self.k - 1):
                rand_ind = np.random.choice(self.X.shape[0], p=dist_arr_sqr / dist_arr_sqr.sum())
                centers_ind.append(rand_ind)
                dist_arr_new = np.linalg.norm(self.X[rand_ind] - self.X, axis=1) ** 2
                dist_arr_sqr = np.minimum(dist_arr_sqr, dist_arr_new)
            self.centers = self.X[centers_ind]

        self.clusters = np.zeros(self.X.shape[0], dtype=int)

        for _ in range(self.max_iter):
            dist = np.array([np.inf] * self.X.shape[0])
            for clust_ind, center in enumerate(self.centers):
                dist_arr = np.linalg.norm(self.X - center, axis=1)
                change_ind = dist_arr < dist
                dist[change_ind] = dist_arr[change_ind]
                self.clusters[change_ind] = clust_ind

            new_centers = np.array([self.X[self.clusters == i].mean(axis=0) if len(self.X[self.clusters == i]) > 0 else self.re_init(self.init_type, dist)
                                     for i in range(self.k)])

            if np.allclose(self.centers, new_centers, atol=1e-4):
                break
            self.centers = new_centers

            # old_centers = self.centers.copy()
            #
            # for cluster in range(self.k):
            #     self.centers[cluster] = self.X[self.clusters == cluster].mean(axis=0)
            # if sum(old_centers == self.centers) == self.k:
            #     break


    
    def predict(self, X: np.array) -> np.array:
        """
        Для каждого элемента из X возвращает номер кластера, 
        к которому относится данный элемент.
        
        Parameters
        ----------
        X : np.array
            Набор данных, для элементов которого находятся ближайшие кластера.
        
        Return
        ------
        labels : np.array
            Вектор индексов ближайших кластеров 
            (по одному индексу для каждого элемента из X).
        
        """
        answer = np.zeros(X.shape[0], dtype=int)
        dist = np.array([np.inf] * X.shape[0])
        for clust_ind, center in enumerate(self.centers):
            dist_arr = np.linalg.norm(center - X, axis=1)
            change_ind = dist_arr < dist
            dist[change_ind] = dist_arr[change_ind]
            answer[change_ind] = clust_ind

        return np.array(answer)
    
# Task 2

class DBScan:
    def __init__(self, eps: float = 0.5, min_samples: int = 5, 
                 leaf_size: int = 40, metric: str = "euclidean"):
        """
        
        Parameters
        ----------
        eps : float, min_samples : int
            Параметры для определения core samples.
            Core samples --- элементы, у которых в eps-окрестности есть 
            хотя бы min_samples других точек.
        metric : str
            Метрика, используемая для вычисления расстояния между двумя точками.
            Один из трех вариантов:
            1. euclidean 
            2. manhattan
            3. chebyshev
        leaf_size : int
            Минимальный размер листа для KDTree.

        """
        self.eps = eps
        self.min_samples = min_samples
        self.leaf_size = leaf_size
        self.metric = metric
        self.X = None
        self.labels = None

    def DFS_rec(self, graph, s, clust_label):
        self.labels[s] = clust_label
        for i in graph[s]:
            if self.labels[i] == -1:
                self.DFS_rec(graph, i, clust_label)

    def DFS(self, graph):
        label = 0
        for i in range(len(graph)):
            if self.labels[i] == -1 and graph[i].size > 0:
                self.DFS_rec(graph, i, label)
                label += 1

    def fit_predict(self, X: np.array, y = None) -> np.array:
        """
        Кластеризует элементы из X, 
        для каждого возвращает индекс соотв. кластера.
        Parameters
        ----------
        X : np.array
            Набор данных, который необходимо кластеризовать.
        y : Ignored
            Не используемый параметр, аналогично sklearn
            (в sklearn считается, что все функции fit_predict обязаны принимать 
            параметры X и y, даже если y не используется).
        Return
        ------
        labels : np.array
            Вектор индексов кластеров
            (Для каждой точки из X индекс соотв. кластера).

        """
        self.X = X
        self.labels = np.zeros(self.X.shape[0], dtype=int) - 1
        kdtree = KDTree(self.X, leaf_size=self.leaf_size, metric=self.metric)
        neighbours = kdtree.query_radius(self.X, r=self.eps)
        graph = [np.array([]) for _ in range(X.shape[0])]
        for i, point_heighb in enumerate(neighbours):
            if point_heighb.size >= self.min_samples:
                graph[i] = point_heighb
        self.DFS(graph)

        return self.labels
# Task 3

class AgglomerativeClustering:
    def __init__(self, n_clusters: int = 16, linkage: str = "average"):
        """

        Parameters
        ----------
        n_clusters : int
            Количество кластеров, которые необходимо найти (то есть, кластеры
            итеративно объединяются, пока их не станет n_clusters)
        linkage : str
            Способ для расчета расстояния между кластерами. Один из 3 вариантов:
            1. average --- среднее расстояние между всеми парами точек,
               где одна принадлежит первому кластеру, а другая - второму.
            2. single --- минимальное из расстояний между всеми парами точек,
               где одна принадлежит первому кластеру, а другая - второму.
            3. complete --- максимальное из расстояний между всеми парами точек,
               где одна принадлежит первому кластеру, а другая - второму.
        """
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.X = None
        self.labels = None

    def fit_predict(self, X: np.array, y=None) -> np.array:
        """
        Кластеризует элементы из X,
        для каждого возвращает индекс соотв. кластера.
        Parameters
        ----------
        X : np.array
            Набор данных, который необходимо кластеризовать.
        y : Ignored
            Не используемый параметр, аналогично sklearn
            (в sklearn считается, что все функции fit_predict обязаны принимать
            параметры X и y, даже если y не используется).
        Return
        ------
        labels : np.array
            Вектор индексов кластеров
            (Для каждой точки из X индекс соотв. кластера).

        """
        self.X = X
        self.labels = np.arange(self.X.shape[0])
        dist_matrix = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=-1)
        np.fill_diagonal(dist_matrix, np.inf)

        while np.unique(self.labels).size > self.n_clusters:
            min_value = np.argmin(dist_matrix)
            clust_join_1, clust_join_2 = np.unravel_index(min_value, dist_matrix.shape)

            clust_1_num = self.labels[self.labels == clust_join_1].size
            clust_2_num = self.labels[self.labels == clust_join_2].size
            self.labels[self.labels == clust_join_2] = clust_join_1

            if self.linkage == 'average':
                for label in np.unique(self.labels):
                    if clust_join_1 != label:
                        dist_matrix[clust_join_1][label] = ((clust_1_num * dist_matrix[clust_join_1][label]
                                                             + clust_2_num * dist_matrix[clust_join_2][label])
                                                            / (clust_1_num + clust_2_num))

                        dist_matrix[label][clust_join_1] = ((clust_1_num * dist_matrix[label][clust_join_1]
                                                             + clust_2_num * dist_matrix[label][clust_join_2])
                                                            / (clust_1_num + clust_2_num))
            elif self.linkage == 'single':
                for label in np.unique(self.labels):
                    if clust_join_1 != label:
                        dist_matrix[clust_join_1][label] = min(dist_matrix[clust_join_1][label],
                                                               dist_matrix[clust_join_2][label])

                        dist_matrix[label][clust_join_1] = min(dist_matrix[label][clust_join_1],
                                                               dist_matrix[label][clust_join_2])
            elif self.linkage == 'complete':
                for label in np.unique(self.labels):
                    if clust_join_1 != label:
                        dist_matrix[clust_join_1][label] = max(dist_matrix[clust_join_1][label],
                                                               dist_matrix[clust_join_2][label])

                        dist_matrix[label][clust_join_1] = max(dist_matrix[label][clust_join_1],
                                                               dist_matrix[label][clust_join_2])

            dist_matrix[:, clust_join_2] = np.inf
            dist_matrix[clust_join_2, :] = np.inf

        _, self.labels = np.unique(self.labels, return_inverse=True)

        return self.labels