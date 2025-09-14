
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

def plot_decision_boundary(clf, X, y, title="Decision Boundary"):
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors="k", s=60)
    plt.title(title)
    plt.show()

def plot_linear_svm_boundary(clf, X, y, title="Linear SVM"):
    if X.shape[1] != 2:
        raise ValueError("Can only visualize 2D datasets")

    w, b = clf.coef_()

    # Create a grid of points
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx = np.linspace(x_min, x_max, 100)

    # Decision boundary: w·x + b = 0  =>  x2 = -(w1/w2)x1 - b/w2
    yy = -(w[0] / w[1]) * xx - b / w[1]
    # Margins: w·x + b = ±1
    yy_plus  = yy + 1 / w[1]
    yy_minus = yy - 1 / w[1]

    plt.figure(figsize=(7, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, s=60, edgecolors="k")
    plt.plot(xx, yy, 'k-', label="Decision boundary")
    plt.plot(xx, yy_plus, 'k--', label="Margin")
    plt.plot(xx, yy_minus, 'k--')

    # Highlight support vectors
    if clf.sv_X is not None:
        plt.scatter(clf.sv_X[:, 0], clf.sv_X[:, 1], 
                    s=150, facecolors='none', edgecolors='red', label="Support Vectors")

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.title(title)
    plt.legend()
    plt.show()


def plot_kernel_svm_boundary(clf, X, y, title="Kernel SVM"):
    if X.shape[1] != 2:
        raise ValueError("Can only visualize 2D datasets")

    # Grid
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # Decision function values for each point in grid
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(7, 6))
    # Contour levels: decision boundary (0), margins (+1/-1)
    plt.contour(xx, yy, Z, levels=[-1, 0, 1],
                linestyles=['--', '-', '--'], colors='k')
    plt.contourf(xx, yy, np.sign(Z), alpha=0.2, cmap=plt.cm.coolwarm)

    # Scatter training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm,
                s=60, edgecolors="k")
    # Highlight support vectors
    if clf.sv_X is not None:
        plt.scatter(clf.sv_X[:, 0], clf.sv_X[:, 1], 
                    s=150, facecolors='none', edgecolors='red', label="Support Vectors")

    plt.title(title)
    plt.legend()
    plt.show()

def plot_classifiers_comparison(classifiers, X, y, titles=None, h=0.02):
    if X.shape[1] != 2:
        raise ValueError("Can only visualize 2D datasets")

    # Mesh grid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    n_clf = len(classifiers)
    fig, axes = plt.subplots(1, n_clf, figsize=(6*n_clf, 5))

    if n_clf == 1:
        axes = [axes]  # make iterable

    for idx, (clf, ax) in enumerate(zip(classifiers, axes)):
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        ax.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, s=60, edgecolors="k")

        # Highlight support vectors for SVM if available
        if hasattr(clf, "sv_X") and clf.sv_X is not None:
            ax.scatter(clf.sv_X[:, 0], clf.sv_X[:, 1], 
                       s=150, facecolors="none", edgecolors="red", label="Support Vectors")
            ax.legend()

        title = titles[idx] if titles is not None else f"Classifier {idx+1}"
        ax.set_title(title)

    plt.show()
