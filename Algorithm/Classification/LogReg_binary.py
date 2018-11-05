import numpy as np


class LogReg_binary:
    """ Binary Logistic classifier
        This class implements regularized binary logistic regression
        and ues Maximum likelihood Estimation for loss function

        Parameter
        ---------
        n_iter: int
        number of iteration Gradient Descent

        learning_rate: float
        alpha in Gradient Descent Algorithm

        reg: float
        lambda in regularization

        report: bool
        showing report of each iter in Gradient Descent

        Return
        ------
        predict values for test data

        Examples
        --------
        >>> clf = LogReg_binary()
        >>> clf.fit(x, y)
        >>> y_pred = clf.predict(x_test)
    """

    def __init__(self, n_iter=1000, learning_rate=0.00001, reg=0.0, report=True):
        self.W = None
        self.lr = learning_rate
        self.n_iter = n_iter
        self.reg = reg
        self.report = report

    @staticmethod
    def simplification(x):
        return np.concatenate((np.ones((x.shape[0], 1)), x), axis=1)

    @staticmethod
    def sigmoid(y):
        return 1 / (1 + np.exp(-y))

    @staticmethod
    def StandardScaler(x):
        return (x - np.mean(x, axis=0)) / np.std(x, axis=0)

    # Optimize Method
    @staticmethod
    def gradient_descent(x, y_pred, y):
        return x.T @ (y_pred - y)

    # Loss Function
    def MLE(self, y, y_pred, w):

        error = -np.sum((y * np.log(y_pred)) + ((1 - y) * np.log(1 - y_pred)))
        return error + (self.reg * (w.T @ w)[0][0])

    def Probability(self, x, theta):
        y_predict = x @ theta
        return self.sigmoid(y_predict)

    def fit(self, x, y):
        # PreProcessing
        x = self.simplification(self.StandardScaler(x))
        y = y.reshape(-1, 1)
        w = 0.001 * np.random.randn(x.shape[1], 1)

        report_time = self.n_iter / 10
        for i in range(self.n_iter):
            # Loss
            y_pred = self.Probability(x, w)
            error = self.MLE(y, y_pred, w)
            # Optimize Weight
            gd = self.gradient_descent(x, y_pred, y)

            w[0] = w[0] - self.lr * gd[0]
            w[1:] = w[1:] - self.lr * (gd[1:] + self.reg * w[1:])
            # Report
            if self.report and i % report_time == 0:
                print('\terror in iter {} = {}'.format(i, error))

        self.W = w

    def predict(self, x_test):
        x = self.simplification(self.StandardScaler(x_test))
        probs = self.Probability(x, self.W)

        y_pred = np.zeros(shape=(x_test.shape[0],), dtype=int)
        y_pred[np.flatnonzero(probs > 0.5)] = 1
        return y_pred

    def score(self, x, y):
        y_pred = self.predict(x)
        return np.mean(y_pred == y)
