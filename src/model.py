import numpy as np

class LinearRegressionGD:

    def __init__(self, n_features=1):
        self.weights = np.zeroes((n_features, 1))
        self.bias = 0
        self.loss_history = []

    def predict(self, X):
        y_pred = self.weights @ X + self.bias
        return y_pred