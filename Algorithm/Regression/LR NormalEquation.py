import numpy as np


class LinerRegressionNE:

    def __init__(self):
        self.theta = None

    @staticmethod
    def normal_equation(x, y):
        # or inv(x.T @ x) @ x.T @ y
        theta = np.linalg.pinv(x)@ y
        return theta

    def fit(self, x, y):
        theta = self.normal_equation(x, y)
        self.theta = theta

    def predict(self, x_test):
        y_predict = np.sum(x_test * self.theta, axis=1)
        return y_predict
