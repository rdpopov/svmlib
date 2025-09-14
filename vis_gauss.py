
from svm import SVM
from kernel import Kernel
from sklearn.datasets import make_moons
from plot_decision_bound import plot_decision_boundary

if __name__ == "__main__":
    X, y = make_moons(n_samples=10, noise=0.2, random_state=42)
    svm = SVM(D=1.0, kernel=Kernel('gauss',{'sigma':1.0}), ).fit(X, y)
    plot_decision_boundary(svm, X, y, title="SVM with Gaussian Kernel")
