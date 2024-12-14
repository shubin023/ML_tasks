import numpy as np

# Task 1

def mse(y_true:np.ndarray, y_predicted:np.ndarray):
    return np.mean((y_true - y_predicted) * (y_true - y_predicted))

def r2(y_true:np.ndarray, y_predicted:np.ndarray):
    return 1 - np.sum((y_true - y_predicted) * (y_true - y_predicted)) / np.sum((y_true - np.mean(y_true)) * (y_true - np.mean(y_true)))

# Task 2

class NormalLR:
    def __init__(self):
        self.weights = None
    
    def fit(self, X:np.ndarray, y:np.ndarray):
        X_1 = np.column_stack((X, np.array([1] * X.shape[0])))
        self.weights = np.linalg.inv(X_1.T @ X_1) @ X_1.T @ y
    
    def predict(self, X:np.ndarray) -> np.ndarray:
        X_1 = np.column_stack((X, np.array([1] * X.shape[0])))
        return X_1 @ self.weights

# Task 3

class GradientLR:
    def __init__(self, alpha:float, iterations=10000, l=0.):
        self.weights = None # Save weights here
    
    def fit(self, X:np.ndarray, y:np.ndarray):
        pass

    def predict(self, X:np.ndarray):
        pass

# Task 4

def get_feature_importance(linear_regression):
    return []

def get_most_important_features(linear_regression):
    return []