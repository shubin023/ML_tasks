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
        X_1 = np.hstack((X, np.ones((X.shape[0], 1))))
        self.weights = np.linalg.inv(X_1.T @ X_1) @ X_1.T @ y
    
    def predict(self, X:np.ndarray) -> np.ndarray:
        X_1 = np.column_stack((X, np.array([1] * X.shape[0])))
        return X_1 @ self.weights

# Task 3

class GradientLR:
    def __init__(self, alpha:float, iterations=10000, l=0.):
        self.weights = None
        self.alpha = alpha
        self.iterations = iterations
        self.l = l
    
    def fit(self, X:np.ndarray, y:np.ndarray):
        X_1 = np.hstack((X, np.ones((X.shape[0], 1))))
        n = X_1.shape[0]

        w = np.random.randn(X_1.shape[1]) * 0.01
        prev_loss = float('inf')
        for iteration in range(self.iterations):
            gradient_part = 2 / n * X_1.T @ (X_1 @ w - y)
            L_1_reg_part = self.l * np.sign(w)
            L_1_reg_part[-1] = 0

            w = w - self.alpha * (gradient_part + L_1_reg_part)

            if iteration % 1000 == 0:
                loss = mse(y, X_1 @ w) + self.l * (np.sum(np.abs(w)) - abs(w[-1]))
                if abs(prev_loss - loss) < 1e-2:
                    break

        self.weights = w

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_1 = np.column_stack((X, np.array([1] * X.shape[0])))
        return X_1 @ self.weights


# Task 4

def get_feature_importance(linear_regression):
    weights_not_bias = linear_regression.weights[:-1]
    return abs(weights_not_bias) / sum(abs(weights_not_bias))

def get_most_important_features(linear_regression):
    imp_arr = sorted([(i, imp) for i, imp in enumerate(get_feature_importance(linear_regression))], key=lambda x: x[1], reverse=True)
    return [x[0] for x in imp_arr]