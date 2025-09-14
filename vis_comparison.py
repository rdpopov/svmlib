from sklearn.datasets import make_classification
from plot_decision_bound import plot_classifiers_comparison

from svm import SVM
from kernel import Kernel
from krr import KLR

if __name__ == "__main__":

    X, y = make_classification(n_samples=200, n_features=2, n_classes=2,
                               n_redundant=0, n_clusters_per_class=1,
                               class_sep=2.0, random_state=420)

    ker = Kernel('poly', {'degree':3, 'gamma':0.5, 'coef0':1.0})

    svm = SVM(D=1.0, kernel= ker).fit(X, y)

    klr = KLR(kernel=ker, lam=1e-2).fit(X, y)
# Compare side by side
    plot_classifiers_comparison(
        classifiers=[svm, klr],
        X=X, y=y,
        titles=["SVM ", "Kernelized Linear Regression"]
    )

