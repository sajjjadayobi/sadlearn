import numpy as np
from Algorithm.preprocessing.normalize import StandardScaler


class LinerRegression:
    """
    Liner Regression:
        Version : Normal Equation
        Ordinary least squares Linear Regression.
        can computing n dimension

    Parameters
    ----------
    standardization : boolean, optional, default False
        if this param was true, data will normalization

    x: parameter in function fit
      x is data to shape of matrix

    y: parameter in function fit
      y or target is correct response for x to shape of vector or matrix

    test: parameter in function predict
        test for Predicting correct response

    Return
    -------
    predictions :  in function predict
    returns values of predicted for test data

    Notes
    -----
    most by import numpy
    if you will predictions just a sample, should doing like this [x[0]]

    Example
    -------
    >>> reg = LinerRegression(standardization=True)
    >>> reg.fit(x, y)
    >>> predictions = reg.predict(x)
    """

    def __init__(self, standardization=False):
        self.intercept = None
        self.coef = None
        self.standardization = standardization

    @staticmethod
    def computing_coef(x, y):
        x_mean = x - np.mean(x)
        y_mean = y - np.mean(y)
        x_m_sq = np.sum(np.square(x_mean))

        m = np.sum(x_mean * y_mean) / x_m_sq
        return m

    def fit(self, x, y):
        if self.standardization:
            # other than import can copy code than preprocessing.normalize
            x = StandardScaler(x)

        coef = []
        for i in x.T:
            coef.append(self.computing_coef(i, y) / len(x.T))
        intercept = np.mean(y) - np.sum(coef) * np.mean(x)

        self.intercept = intercept
        self.coef = coef

    def predict(self, test):
        if self.standardization:
            test = StandardScaler(test)

        predictions = []
        for x in test:
            slop = []
            for i, v in enumerate(x):
                slop.append(v * self.coef[i])

            predictions.append(np.sum(slop) + self.intercept)

        return predictions
