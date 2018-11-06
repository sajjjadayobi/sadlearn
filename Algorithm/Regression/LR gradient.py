import numpy as np
import matplotlib.pyplot as plt


class LinReg_gradient_descent:

    """
    Linear Regression with gradient descent

    Parameter
    ---------
    n_iter: int
    number of iteration Gradient Descent

    learning_rate: float
    alpha in Gradient Descent Algorithm

    reg: float
    coef of regularization

    report: bool
    showing report of each iter in Gradient Descent


    Examples
    --------
    >>> clf = LinReg_gradient_descent()
    >>> clf.fit(x, y)
    >>> y_pred = clf.predict(x_test)
    """

    def __init__(self, n_iter=1000, learning_rate=0.00001, reg=0.0, report=True):
        self.weight = None
        self.lr = learning_rate
        self.n_iter = n_iter
        self.report = report
        self.reg = reg

    #  x = x0 + x added bias
    @staticmethod
    def simplification(x):
        return np.concatenate((np.ones((x.shape[0], 1)), x), axis=1)

    # Normalize data
    @staticmethod
    def StandardScaler(x):
        return (x - np.mean(x, axis=0)) / np.std(x, axis=0)

    # Optimize Method
    @staticmethod
    def gradient_descent(x, y_pred, y):
        return x.T @ (y_pred - y)

    # Loss Function
    def sum_square_error(self, y, y_pred, w):
        error = y_pred - y
        loss = ((error.T @ error) * 0.5)[0][0]
        # added Regularize
        return loss + (self.reg * (w.T @ w)[0][0])

    def fit(self, x, y):
        # PreProcessing
        x = self.simplification(self.StandardScaler(x))
        y = y.reshape(-1, 1)
        w = 0.001 * np.random.randn(x.shape[1], 1)

        report_time = self.n_iter / 10
        for i in range(self.n_iter):
            # Loss
            y_pred = x @ w
            error = self.sum_square_error(y, y_pred, w)
            # Optimize Weight
            gd = self.gradient_descent(x, y_pred, y)
            w[0] = w[0] - (self.lr * gd[0])
            w[1:] = w[1:] - (self.lr * (gd[1:] + (self.reg * w[1:])))

            # Report
            if self.report and i % report_time == 0:
                print('\t error in iter {} = {}'.format(i, error))

        self.weight = w

    def predict(self, x_test):
        x = self.simplification(self.StandardScaler(x_test))
        return x @ self.weight
