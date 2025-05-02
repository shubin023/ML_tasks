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
        answer = self.w @ X_new.T
        answer = np.where(answer >= 0, 1, -1)
        predictions = np.where(answer == -1, self.labels[0], self.labels[1])
        return predictions

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
        for _ in range(self.iterations - 10000):
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
        answer = self.w @ X_new.T
        answer = np.where(answer >= 0, 1, -1)
        predictions = np.where(answer == -1, self.labels[0], self.labels[1])
        return predictions


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

    n, h, w = images.shape
    half_h = h // 2
    half_w = w // 2

    h_sym = np.zeros(n, dtype=float)
    v_sym = np.zeros(n, dtype=float)

    for i in range(n):
        img = images[i]

        # Горизонтальная симметрия
        top_half = img[:half_h, :]
        bottom_half = img[half_h:, :]

        bottom_half_flipped = np.flipud(bottom_half)

        min_h = min(top_half.shape[0], bottom_half_flipped.shape[0])
        top_crop = top_half[:min_h, :]
        bottom_crop = bottom_half_flipped[:min_h, :]

        diff_h = np.abs(top_crop - bottom_crop)
        sum_h = np.abs(top_crop) + np.abs(bottom_crop)

        h_sym[i] = 1 - (np.sum(diff_h) / np.sum(sum_h) + 1e-8)

        # Вертикальная симметрия
        left_half = img[:, :half_w]
        right_half = img[:, half_w:]

        right_half_flipped = np.fliplr(right_half)

        min_w = min(left_half.shape[1], right_half_flipped.shape[1])
        left_crop = left_half[:, :min_w]
        right_crop = right_half_flipped[:, :min_w]

        diff_v = np.abs(left_crop - right_crop)
        sum_v = np.abs(left_crop) + np.abs(right_crop)

        v_sym[i] = 1 - (np.sum(diff_v) / np.sum(sum_v) + 1e-8)

    raw_features = np.column_stack((h_sym, v_sym))

    mean = raw_features.mean(axis=0)
    std = raw_features.std(axis=0)

    scaled_features = (raw_features - mean) / (std + 1e-8)

    return scaled_features
