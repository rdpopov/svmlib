import numpy as np

def LinearKernel(X,Z,parameters):
    return X @ Z.T

def GaussKernel(X,Z,parameters):
    sigma = parameters.get('sigma', 1.0 / X.shape[1])
    X_norm = np.sum(X*X, axis=1)[:, None]
    Z_norm = np.sum(Z*Z, axis=1)[None, :]
    sqdist = X_norm + Z_norm - 2.0 * (X @ Z.T)
    return np.exp(-sigma  * np.clip(sqdist, 0.0, None))

def PolynomialKernel(X,Z,parameters):
    sigma = parameters.get('sigma', 1.0 / X.shape[1])
    degree = parameters.get('degree', 3)
    bias = parameters.get('bias', 0.0)

    return (sigma * (X @ Z.T) + bias) ** degree

class Kernel:
    def __init__(self, kernelType, parameters: dict = {}):
        self.kType = kernelType
        self.parameters = parameters
        self.tested_kernel_with_rules = None

    def SymetricalTest(self,X):
        K1 = self.execute_unsafe(X)
        K2 = self.execute_unsafe(X).T
        return np.allclose(K1,K2)

    def MercerTest(self,X):
        K = self.execute_unsafe(X)
        eigvals = np.linalg.eigvalsh(K)
        return np.all(eigvals >= -1e-8)

    def IsLinear(self):
        return self.kType == 'linear'

    def execute_unsafe(self,X,Z=None):
        if Z is None:
            Z = X
        if callable(self.kType ):
            return self.kType (X, Z)
        if self.kType  == 'linear':
            return LinearKernel(X,Z,self.parameters)
        if self.kType  == 'gauss':
            return GaussKernel(X,Z,self.parameters)
        if self.kType  == 'poly':
            return PolynomialKernel(X,Z,self.parameters)
        raise ValueError(f"Unknown kernel: {self.kType }")

    def __call__(self,X,Z=None):
        if self.tested_kernel_with_rules is None:
            self.tested_kernel_with_rules = False
            try:
                if not self.SymetricalTest(X):
                    Exception("Warning: Kernel is not symetrical")
                if not self.MercerTest(X):
                    print("Warning: Kernel does not satisfy Mercer's condition")
            except Exception as e:
                print(f"Warning: Kernel test failed with error: {e}")

            self.tested_kernel_with_rules = True
        elif self.tested_kernel_with_rules == False:
            Exception("Kernel has failed testing")

        if self.tested_kernel_with_rules == True:
            return self.execute_unsafe(X,Z)

if __name__ == "__main__":
    # Example usage and testing
    print("Testing Kernels")
    X = np.array([[1, 2], [3, 4], [5, 6]])
    kernel = Kernel('linear')
    print("Linear Kernel:\n", kernel(X))

    kernel_rbf = Kernel('gauss', {'sigma': 0.5})
    print("Gaussian Kernel:\n", kernel_rbf(X))

    kernel_poly = Kernel('poly', {'degree': 2, 'gamma': 1, 'bias': 1})
    print("Polynomial Kernel:\n", kernel_poly(X))
