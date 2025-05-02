import numpy as np
import copy
from typing import List, NoReturn
import torch
from torch import nn
import torch.nn.functional as F


# Task 1

class Module:
    """
    Абстрактный класс. Его менять не нужно. Он описывает общий интерфейс взаимодествия со слоями нейронной сети.
    """
    def forward(self, x):
        pass
    
    def backward(self, d):
        pass
        
    def update(self, alpha):
        pass
    
    
class Linear(Module):
    """
    Линейный полносвязный слой.
    """
    def __init__(self, in_features: int, out_features: int):
        """
        Parameters
        ----------
        in_features : int
            Размер входа.
        out_features : int 
            Размер выхода.
    
        Notes
        -----
        W и b инициализируются случайно.
        """
        self.W = np.random.normal(0, 1 / np.sqrt((in_features + 1) * (in_features + 1) + out_features * out_features), (in_features + 1, out_features))
        self.x = None
        self.dW = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Возвращает y = Wx + b.

        Parameters
        ----------
        x : np.ndarray
            Входной вектор или батч.
            То есть, либо x вектор с in_features элементов,
            либо матрица размерности (batch_size, in_features).
    
        Return
        ------
        y : np.ndarray
            Выход после слоя.
            Либо вектор с out_features элементами,
            либо матрица размерности (batch_size, out_features)

        """
        if x.ndim == 1:
            self.x = np.append(x, 1)
        else:
            self.x = np.hstack((x, np.ones((x.shape[0], 1))))
        return self.x @ self.W
    
    def backward(self, d: np.ndarray) -> np.ndarray:
        """
        Cчитает градиент при помощи обратного распространения ошибки.

        Parameters
        ----------
        d : np.ndarray
            Градиент.
        Return
        ------
        np.ndarray
            Новое значение градиента.
        """
        if self.x.ndim == 1:
            self.dW = self.x.reshape(-1, 1) @ d.reshape(1, -1)
        else:
            self.dW = self.x.T @ d
        return d @ self.W[:-1].T
        
    def update(self, alpha: float) -> NoReturn:
        """
        Обновляет W и b с заданной скоростью обучения.

        Parameters
        ----------
        alpha : float
            Скорость обучения.
        """
        self.W -= alpha * self.dW
    

class ReLU(Module):
    """
    Слой, соответствующий функции активации ReLU. Данная функция возвращает новый массив, в котором значения меньшие 0 заменены на 0.
    """
    def __init__(self):
        self.x = None
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Возвращает y = max(0, x).

        Parameters
        ----------
        x : np.ndarray
            Входной вектор или батч.
    
        Return
        ------
        y : np.ndarray
            Выход после слоя (той же размерности, что и вход).

        """
        self.x = x
        return np.maximum(0, x)
        
    def backward(self, d) -> np.ndarray:
        """
        Cчитает градиент при помощи обратного распространения ошибки.

        Parameters
        ----------
        d : np.ndarray
            Градиент.
        Return
        ------
        np.ndarray
            Новое значение градиента.
        """
        return d * (self.x > 0)


class SoftMax(Module):
    """
    Слой, соответствующий функции активации SoftMax. Данная функция возвращает новый массив со значениями от 0 до 1.
    """

    def __init__(self):
        self.y = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        x : np.ndarray
            Входной вектор или батч.

        Return
        ------
        y : np.ndarray
            Выход после слоя (той же размерности, что и вход).
        """
        self.y = np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
        return self.y

    def backward(self, d) -> np.ndarray:
        """
        Cчитает градиент при помощи обратного распространения ошибки.

        Parameters
        ----------
        d : np.ndarray
            Градиент.
        Return
        ------
        np.ndarray
            Новое значение градиента.
        """

        # return self.y * (d - np.sum(d * self.y, axis=1, keepdims=True))
        return d

# Task 2

class MLPClassifier:
    def __init__(self, modules: List[Module], epochs: int = 40, alpha: float = 0.01, batch_size: int = 32):
        """
        Parameters
        ----------
        modules : List[Module]
            Cписок, состоящий из ранее реализованных модулей и 
            описывающий слои нейронной сети. 
            В конец необходимо добавить Softmax.
        epochs : int
            Количество эпох обучения.
        alpha : float
            Cкорость обучения.
        batch_size : int
            Размер батча, используемый в процессе обучения.
        """
        self.layers = modules.copy()
        self.layers.append(SoftMax())
        self.epochs = epochs
        self.alpha = alpha
        self.batch_size = batch_size
        self.loss = None
            
    def fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Обучает нейронную сеть заданное число эпох. 
        В каждой эпохе необходимо использовать cross-entropy loss для обучения, 
        а так же производить обновления не по одному элементу, а используя батчи (иначе обучение будет нестабильным и полученные результаты будут плохими.

        Parameters
        ----------
        X : np.ndarray
            Данные для обучения.
        y : np.ndarray
            Вектор меток классов для данных.
        """
        n_samples = X.shape[0]

        for _ in range(self.epochs):
            indices = np.arange(n_samples)
            np.random.shuffle(indices)

            for start in range(0, n_samples, self.batch_size):
                end = start + self.batch_size
                batch_ind = indices[start:end]
                X_batch = X[batch_ind]
                y_batch = y[batch_ind]

                for layer in self.layers:
                    X_batch = layer.forward(X_batch)

                grad = X_batch.copy()
                grad[np.arange(len(y_batch)), y_batch] -= 1

                for layer in reversed(self.layers):
                    grad = layer.backward(grad)

                    if hasattr(layer, 'update'):
                        layer.update(self.alpha)
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Предсказывает вероятности классов для элементов X.

        Parameters
        ----------
        X : np.ndarray
            Данные для предсказания.
        
        Return
        ------
        np.ndarray
            Предсказанные вероятности классов для всех элементов X.
            Размерность (X.shape[0], n_classes)
        
        """
        for layer in self.layers:
            X = layer.forward(X)

        return X
        
    def predict(self, X) -> np.ndarray:
        """
        Предсказывает метки классов для элементов X.

        Parameters
        ----------
        X : np.ndarray
            Данные для предсказания.
        
        Return
        ------
        np.ndarray
            Вектор предсказанных классов
        
        """
        p = self.predict_proba(X)
        return np.argmax(p, axis=1)


# Task 3

classifier_moons = MLPClassifier([
    Linear(2, 8),
    ReLU(),
    Linear(8, 8),
    ReLU(),
    Linear(8, 2)
])
classifier_blobs = MLPClassifier([
    Linear(2, 8),
    ReLU(),
    Linear(8, 8),
    ReLU(),
    Linear(8, 3)
])


# Task 4

class TorchModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(3 * 32 * 32, 1024)
        self.l2 = nn.Linear(1024, 512)
        self.l3 = nn.Linear(512, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)

        x = self.l1(x)
        x = F.relu(x)

        x = self.l2(x)
        x = F.relu(x)

        x_res = self.l3(x)

        probs = F.softmax(x_res, dim=1)
        return probs

    def load_model(self):
        """
        Используйте torch.load, чтобы загрузить обученную модель
        Учтите, что файлы решения находятся не в корне директории, поэтому необходимо использовать следующий путь:
        `__file__[:-7] + "model.pth"`, где "model.pth" - имя файла сохраненной модели `
        """
        model_path = __file__[:-7] + "model.pth"
        self.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    
    def save_model(self):
        """
        Используйте torch.save, чтобы сохранить обученную модель
        """
        model_path = __file__[:-7] + "model.pth"
        torch.save(self.state_dict(), model_path)
        
def calculate_loss(X: torch.Tensor, y: torch.Tensor, model: TorchModel):
    """
    Cчитает cross-entropy.

    Parameters
    ----------
    X : torch.Tensor
        Данные для обучения.
    y : torch.Tensor
        Метки классов.
    model : Model
        Модель, которую будем обучать.

    """
    probs = model(X)
    log_probs = torch.log(probs[range(len(y)), y])
    loss = -torch.sum(log_probs)
    return loss
