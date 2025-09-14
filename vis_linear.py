
import numpy as np
from kernel import Kernel
from svm import SVM
from plot_decision_bound import plot_linear_svm_boundary
from sklearn.datasets import make_blobs


if __name__ == "__main__":
    X, y = make_blobs(n_samples=50, n_features=2, centers=2, cluster_std=1.05)
    y = np.where(y == 0, -1, 1)
    svm_linear = SVM(D=1.0, kernel=Kernel("none")).fit(X, y)

# Plot
    plot_linear_svm_boundary(svm_linear, X, y, title="Linear SVM Decision Boundary")
