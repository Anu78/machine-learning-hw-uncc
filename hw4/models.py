import numpy as np

class SVM():
    def __init__(self, learningRate=0.01, iterations=1000, lambdaParam=0.01):
        self.lr = learningRate
        self.iter = iterations
        self.w = None
        self.b = None
        self.lp = lambdaParam
    
    def fit(self, X, y):
        _, features = X.shape
        self.w = np.zeros(features)
        self.b = 0

        for _ in range(self.iter):
            for index, xI in enumerate(X):
                condition = y[index] * (np.dot(xI, self.w) - self.b) >= 1

                if condition:
                    self.w -= self.lr * (2 * self.lp * self.w)
                else:
                    self.w -= self.lr * (2 * self.lp * self.w - np.dot(xI, y[index]))
                    self.b -= self.lr * y[index]
    def predict(self, X):
        approx = np.dot(X, self.w) - self.b

        return np.sign(approx)