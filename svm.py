import numpy as np
from kernel import Kernel
from sklearn.datasets import make_blobs

class SVM:
    def __init__(self,
                 D=1.0,
                 kernel=Kernel('linear'),
                 tol=1e-3,
                 max_passes=5):
        # Margin parameter
        self.D = float(D) 
        # Kernel function
        self.kernel = kernel

        self.tol = tol
        self.max_passes = max_passes

        self.LM = None
        self.bias = 0.0
        self.X = None
        self.y = None
        self._K = None

        self.sv_idx = None
        self.sv_X = None
        self.sv_y = None
        self.sv_LM = None

    def _relation(self, i, j):
        return 2.0 * self._K[i, j] - self._K[i, i] - self._K[j, j]

    def _compute_bounds(self, i, j):
        if self.y[i] != self.y[j]:
            L = max(0.0, self.LM[j] - self.LM[i])
            H = min(self.D, self.D + self.LM[j] - self.LM[i])
        else:
            L = max(0.0, self.LM[i] + self.LM[j] - self.D)
            H = min(self.D, self.LM[i] + self.LM[j])
        return L, H

    def _update_bias(self, i, j, Error_i, Error_j, LM_i_old, LM_j_old):
        b1 = (self.bias - Error_i
              - self.y[i]*(self.LM[i]-LM_i_old)*self._K[i, i]
              - self.y[j]*(self.LM[j]-LM_j_old)*self._K[i, j])
        b2 = (self.bias - Error_j
              - self.y[i]*(self.LM[i]-LM_i_old)*self._K[i, j]
              - self.y[j]*(self.LM[j]-LM_j_old)*self._K[j, j])
        if 0 < self.LM[i] < self.D:
            return b1
        if 0 < self.LM[j] < self.D:
            return b2
        return 0.5 * (b1 + b2)

    def _decision_idx(self, i):
        return np.sum(self.LM * self.y * self._K[:, i]) + self.bias

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y, float)
        y = np.where(y <= 0, -1.0, 1.0)
        n_samples = X.shape[0]

        self.X, self.y = X, y
        self.LM = np.zeros(n_samples)
        self.bias = 0.0
        self._K = self.kernel(X)

        passes = 0
        while passes < self.max_passes:
            LM_changed = 0
            for i in range(n_samples):
                Error_i = self._decision_idx(i) - y[i]
                if ((y[i]*Error_i < -self.tol and self.LM[i] < self.D) or
                    (y[i]*Error_i >  self.tol and self.LM[i] > 0)):

                    j = self._pick_random_index(i, n_samples)
                    Error_j = self._decision_idx(j) - y[j]
                    LM_i_old, LM_j_old = self.LM[i], self.LM[j]

                    L, H = self._compute_bounds(i, j)
                    if L == H:
                        continue

                    eta = self._relation(i, j)
                    if eta >= 0:
                        continue

                    self.LM[j] -= y[j] * (Error_i - Error_j) / eta
                    self.LM[j] = np.clip(self.LM[j], L, H)

                    if abs(self.LM[j] - LM_j_old) < 1e-7:
                        continue

                    self.LM[i] += y[i]*y[j]*(LM_j_old - self.LM[j])
                    self.bias = self._update_bias(i, j, Error_i, Error_j, LM_i_old, LM_j_old)

                    LM_changed += 1

            passes = passes + 1 if LM_changed == 0 else 0

        self._cache_support_vectors()
        return self

    def _pick_random_index(self, i, n_samples):
        return np.random.choice([x for x in range(n_samples) if x != i])

    def _cache_support_vectors(self):
        sv_mask = self.LM > 1e-6
        self.sv_idx = np.where(sv_mask)[0]
        self.sv_X = self.X[sv_mask]
        self.sv_y = self.y[sv_mask]
        self.sv_LM = self.LM[sv_mask]

    # ------------------------- Prediction -------------------------
    def decision_function(self, X):
        X = np.asarray(X, float)
        if self.sv_X is None or len(self.sv_X) == 0:
            return np.zeros(X.shape[0]) + self.bias
        K = self.kernel(X, self.sv_X)
        return (K @ (self.sv_LM * self.sv_y)) + self.bias

    def predict(self, X):
        return np.sign(self.decision_function(X))

    def coef_(self):
        if not self.kernel.IsLinear():
            raise AttributeError("coef_ is only available for linear kernel")
        return (self.sv_LM * self.sv_y) @ self.sv_X, self.bias

if __name__ == "__main__":
    X, y = make_blobs(n_samples=50, n_features=2, centers=2, cluster_std=1.05)
    y = np.where(y == 0, -1, 1)

    svm = SVM(D=1.0, kernel=Kernel('linear'), max_passes=10)
    svm.fit(X, y)
    X_test = np.array([[0, 0], [3, 3], [6, 6]])
    y_pred = svm.predict(X_test)
    print("Predictions:", y_pred)

    svm_linear = SVM(D=1.0, kernel=Kernel('linear')).fit(X, y)
    w, b = svm_linear.coef_()
    print("Weights:", w, "Bias:", b)

