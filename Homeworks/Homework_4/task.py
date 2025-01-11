import numpy as np
import copy
from typing import NoReturn


# Task 1

class Perceptron:
    def __init__(self, iterations: int = 100):
        """
        Parameters
        ----------
        iterations : int
        Количество итераций обучения перцептрона.

        Attributes
        ----------
        w : np.ndarray
        Веса перцептрона размерности X.shape[1] + 1 (X --- данные для обучения), 
        w[0] должен соответстовать константе, 
        w[1:] - коэффициентам компонент элемента X.

        Notes
        -----
        Вы можете добавлять свои поля в класс.
        
        """
        self.w = None
        self.labels = None
        self.iterations = iterations
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Обучает простой перцептрон. 
        Для этого сначала инициализирует веса перцептрона,
        а затем обновляет их в течении iterations итераций.
        
        Parameters
        ----------
        X : np.ndarray
            Набор данных, на котором обучается перцептрон.
        y: np.ndarray
            Набор меток классов для данных.
        
        """
        y_true = y.copy()
        self.labels = np.unique(y_true)
        y_true[y_true == self.labels[0]] = -1
        y_true[y_true == self.labels[1]] = 1
        X_new = np.hstack((np.ones((X.shape[0], 1)), X))
        self.w = np.zeros(X_new.shape[1], dtype=float)
        for _ in range(self.iterations):
            margins = (self.w @ X_new.T) * y_true
            wrong_labels = margins <= 0
            self.w += y_true[wrong_labels] @ X_new[wrong_labels]
            
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Предсказывает метки классов.
        
        Parameters
        ----------
        X : np.ndarray
            Набор данных, для которого необходимо вернуть метки классов.
        
        Return
        ------
        labels : np.ndarray
            Вектор индексов классов 
            (по одной метке для каждого элемента из X).
        
        """
        X_new = np.hstack((np.ones((X.shape[0], 1)), X))
        answer = np.sign(self.w @ X_new.T)
        answer[answer == -1] = self.labels[0]
        answer[answer == 1] = self.labels[1]
        return answer
    
# Task 2

class PerceptronBest:

    def __init__(self, iterations: int = 100):
        """
        Parameters
        ----------
        iterations : int
        Количество итераций обучения перцептрона.

        Attributes
        ----------
        w : np.ndarray
        Веса перцептрона размерности X.shape[1] + 1 (X --- данные для обучения), 
        w[0] должен соответстовать константе, 
        w[1:] - коэффициентам компонент элемента X.

        Notes
        -----
        Вы можете добавлять свои поля в класс.
        
        """
        self.w = None
        self.labels = None
        self.iterations = iterations
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Обучает перцептрон.

        Для этого сначала инициализирует веса перцептрона, 
        а затем обновляет их в течении iterations итераций.

        При этом в конце обучения оставляет веса, 
        при которых значение accuracy было наибольшим.
        
        Parameters
        ----------
        X : np.ndarray
            Набор данных, на котором обучается перцептрон.
        y: np.ndarray
            Набор меток классов для данных.
        
        """
        y_true = y.copy()
        self.labels = np.unique(y_true)
        y_true[y_true == np.unique(y_true)[0]] = -1
        y_true[y_true == np.unique(y_true)[1]] = 1
        X_new = np.hstack((np.ones((X.shape[0], 1)), X))
        self.w = np.zeros(X_new.shape[1], dtype=float)
        wrong_labels_cnt = X_new.shape[0]
        best_w = np.copy(self.w)
        for _ in range(self.iterations):
            margins = (self.w @ X_new.T) * y_true
            wrong_labels = margins <= 0
            if wrong_labels.sum() < wrong_labels_cnt:
                best_w = np.copy(self.w)
                wrong_labels_cnt = wrong_labels.sum()
            self.w += y_true[wrong_labels] @ X_new[wrong_labels]
        margins = (self.w @ X_new.T) * y_true
        wrong_labels = margins <= 0
        if wrong_labels.sum() < wrong_labels_cnt:
            best_w = np.copy(self.w)
        self.w = np.copy(best_w)

            
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Предсказывает метки классов.
        
        Parameters
        ----------
        X : np.ndarray
            Набор данных, для которого необходимо вернуть метки классов.
        
        Return
        ------
        labels : np.ndarray
            Вектор индексов классов 
            (по одной метке для каждого элемента из X).
        
        """
        X_new = np.hstack((np.ones((X.shape[0], 1)), X))
        answer = np.sign(self.w @ X_new.T)
        answer[answer == -1] = self.labels[0]
        answer[answer == 1] = self.labels[1]
        return answer
    
# Task 3

def transform_images(images: np.ndarray) -> np.ndarray:
    """
    Переводит каждое изображение в вектор из двух элементов.
        
    Parameters
    ----------
    images : np.ndarray
        Трехмерная матрица с черное-белыми изображениями.
        Её размерность: (n_images, image_height, image_width).

    Return
    ------
    np.ndarray
        Двумерная матрица с преобразованными изображениями.
        Её размерность: (n_images, 2).
    """
    top_half = images[:, :images.shape[1] // 2, :]
    bottom_half = images[:, images.shape[1] // 2:, :]
    brightness_gradient = np.mean(top_half, axis=(1, 2)) - np.mean(bottom_half, axis=(1, 2))

    n_images, height, width = images.shape
    pixel_positions = np.arange(height).reshape(-1, 1)
    vertical_center_of_mass = np.sum(images * pixel_positions, axis=(1, 2)) / np.sum(images, axis=(1, 2))

    # width = images.shape[2]
    # pixel_positions = np.arange(width)
    # horizontal_center_of_mass = np.sum(images * pixel_positions, axis=(1, 2)) / np.sum(images, axis=(1, 2))

    return np.column_stack((brightness_gradient, vertical_center_of_mass))
