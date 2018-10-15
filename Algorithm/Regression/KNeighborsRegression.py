import numpy as np


class KNeighborsRegressor:
    """
    KNeighborsRegressor.py
    detail: the algorithm for Regression with use Nearest neighbors
    Regression based on k-nearest neighbors.
    https://en.wikipedia.org/wiki/K-nearest_neighbor_algorithm

    Parameters
    ----------
    :param x: train data for learning
    :param y: correct response of train data
    :param tests in predict: test data for validation and predict
    :param n_neighbors: number of neighbors

    :return: predict values for test data

    Examples
    --------
    >>> clf = KNeighborsRegressor(x, y, 5)
    >>> y_pred = clf.predict(tests)
    """
    def __init__(self, x, y, n_neighbors):
        self.k = n_neighbors
        self.x = x
        self.y = y

    @staticmethod
    def ocliden(x, y):
        total = 0
        for j in range(len(y)):
            total += np.sqrt((x[j] - y[j]) ** 2)
        return total

    def predict(self, tests):

        pred = np.empty(0)
        for test in tests:
            distance = np.empty(0)
            # get all distance between test_ & train_s
            for j in range(len(self.x)):
                distance = np.append(distance, self.ocliden(self.x[j], test))

            min_dist = []
            for j in range(self.k):
                index = np.where(distance == np.min(distance))[0][0]
                min_dist.append(self.y[index])
                # delete min
                distance = np.delete(distance, [index])

            pred = np.append(pred, np.mean(min_dist))

        return pred[:, np.newaxis]

