import numpy as np
from kernel import Kernel

class KLR:
    def __init__(self, kernel=Kernel('linear'), lam=1e-3):
        self.kernel = kernel
        self.lam = lam
        
        self.X_train = None
        self.y_train = None
        self.alpha = None
        self.bias = 0.0
        self._K = None

    def fit(self, X, y):
        X = np.asarray(X, float)
        y = np.asarray(y, float)
        y = np.where(y <= 0, -1.0, 1.0)
        self.X_train = X
        self.y_train = y
        
        self._K = self.kernel(X)
        n_samples = X.shape[0]
        
        self.alpha = np.linalg.solve(self._K + self.lam * np.eye(n_samples), y)
        self.bias = 0.0
        return self
    
    def decision_function(self, X):
        X = np.asarray(X, float)
        K_test = self.kernel(X, self.X_train)
        return K_test @ self.alpha + self.bias
    
    def predict(self, X):
        return np.sign(self.decision_function(X))
