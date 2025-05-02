import numpy as np
import copy
from cvxopt import spmatrix, matrix, solvers
from sklearn.datasets import make_classification, make_moons, make_blobs
from typing import NoReturn, Callable

solvers.options['show_progress'] = False

# Task 1

class LinearSVM:
    def __init__(self, C: float):
        """
        
        Parameters
        ----------
        C : float
            Soft margin coefficient.
        
        """
        self.C = C
        self.w = None
        self.b = None
        self.support = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Обучает SVM, решая задачу оптимизации при помощи cvxopt.solvers.qp
        
        Parameters
        ----------
        X : np.ndarray
            Данные для обучения SVM.
        y : np.ndarray
            Бинарные метки классов для элементов X 
            (можно считать, что равны -1 или 1). 
        
        """
        n = X.shape[0]

        kernel = X @ X.T

        P = matrix(np.outer(y, y) * kernel)
        q = matrix(-np.ones(n))

        G = matrix(np.vstack((-np.eye(n), np.eye(n))))
        h = matrix(np.hstack((np.zeros(n), np.ones(n) * self.C)))

        A = matrix(y.astype(float), (1, n))
        b = matrix(0.0)

        solution = solvers.qp(P, q, G, h, A, b)

        alphas = np.array(solution['x']).flatten()

        self.w = np.sum((alphas * y)[:, None] * X, axis=0)
        self.support = np.where((alphas > 1e-4) & (alphas < self.C))[0]
        self.b = np.mean(X[self.support] @ self.w - y[self.support])

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Возвращает значение решающей функции.
        
        Parameters
        ----------
        X : np.ndarray
            Данные, для которых нужно посчитать значение решающей функции.

        Return
        ------
        np.ndarray
            Значение решающей функции для каждого элемента X 
            (т.е. то число, от которого берем знак с целью узнать класс).     
        
        """
        return X @ self.w - self.b

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Классифицирует элементы X.
        
        Parameters
        ----------
        X : np.ndarray
            Данные, которые нужно классифицировать

        Return
        ------
        np.ndarray
            Метка класса для каждого элемента X.   
        
        """
        return np.sign(self.decision_function(X))
    
# Task 2

# def get_polynomial_kernel(c=1, power=2):
#     "Возвращает полиномиальное ядро с заданной константой и степенью"
#     def kernel(X, y):
#         return (X @ y + c) ** power
#
#     return kernel
#
# def get_gaussian_kernel(sigma=1.):
#     "Возвращает ядро Гаусса с заданным коэффицинтом сигма"
#     def kernel(X, y):
#         return np.exp(-sigma * (np.linalg.norm(X - y, axis=1) ** 2))
#
#     return kernel

def get_polynomial_kernel(c=1, power=2):
    "Возвращает полиномиальное ядро с заданной константой и степенью"
    def kernel(X, Y):
        return (X @ Y.T + c) ** power

    return kernel

def get_gaussian_kernel(sigma=1.):
    "Возвращает ядро Гаусса с заданным коэффицинтом сигма"
    def kernel(X, Y):
        X_norm = np.sum(X ** 2, axis=1)[:, np.newaxis]
        Y_norm = np.sum(Y ** 2, axis=1)[np.newaxis, :]
        dists = X_norm + Y_norm - 2 * X @ Y.T
        return np.exp(-sigma * dists)
    return kernel


# Task 3

class KernelSVM:
    def __init__(self, C: float, kernel: Callable):
        """
        
        Parameters
        ----------
        C : float
            Soft margin coefficient.
        kernel : Callable
            Функция ядра.
        
        """
        self.C = C
        self.kernel = kernel
        self.alphas = None
        self.support = None
        self.b = None
        self.X = None
        self.y = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Обучает SVM, решая задачу оптимизации при помощи cvxopt.solvers.qp
        
        Parameters
        ----------
        X : np.ndarray
            Данные для обучения SVM.
        y : np.ndarray
            Бинарные метки классов для элементов X 
            (можно считать, что равны -1 или 1). 
        
        """
        n = X.shape[0]

        K = self.kernel(X, X)

        P = matrix(np.outer(y, y) * K)
        q = matrix(-np.ones(n))

        G = matrix(np.vstack((-np.eye(n), np.eye(n))))
        h = matrix(np.hstack((np.zeros(n), np.ones(n) * self.C)))

        A = matrix(y.astype(float), (1, n))
        b = matrix(0.0)

        solution = solvers.qp(P, q, G, h, A, b)

        self.alphas = np.array(solution['x']).flatten()

        self.support = np.where((self.alphas > 1e-8) & (self.alphas < self.C))[0]
        self.X = X
        self.y = y
        self.b = np.mean(np.dot(self.alphas * y, K[:, self.support]) - y[self.support])

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Возвращает значение решающей функции.
        
        Parameters
        ----------
        X : np.ndarray
            Данные, для которых нужно посчитать значение решающей функции.

        Return
        ------
        np.ndarray
            Значение решающей функции для каждого элемента X 
            (т.е. то число, от которого берем знак с целью узнать класс).     
        
        """
        support = np.where(self.alphas > 1e-8)[0]
        K_test = self.kernel(self.X[support], X)
        return np.dot(self.alphas[support] * self.y[support], K_test) - self.b

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Классифицирует элементы X.
        
        Parameters
        ----------
        X : np.ndarray
            Данные, которые нужно классифицировать

        Return
        ------
        np.ndarray
            Метка класса для каждого элемента X.   
        
        """
        return np.sign(self.decision_function(X))
