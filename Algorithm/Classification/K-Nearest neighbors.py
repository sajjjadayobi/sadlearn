import numpy as np

"""
KNearestNeighbors.py
detail: Classifier Algorithm 

author: sajjad ayobi
see others in repository : sadlearn
in URL: https://github.com/sajjjadayobi/sadlearn/
"""


class KNearestNeighbors:
    """
    K-Nearest Neighbors: the algorithm for classification with use Nearest neighbors
    https://en.wikipedia.org/wiki/K-nearest_neighbor_algorithm

    Parameters
    ----------
    :param x: train data for learning
    :param tests in predict: test data for validation and predict
    :param y: correct response of train data
    :param n_neighbors: number of neighbors

    :return: predict values for test data

    Examples
    --------
    >>> clf = KNearestNeighbors(x, y, 5)
    >>> y_pred = clf.predict(tests)
    """
    def __init__(self, x, y, n_neighbors):
        self.x = x
        self.y = y
        self.k = n_neighbors

    @staticmethod
    def ocliden(x, y):
        total = 0
        for j in range(len(y)):
            total += np.sqrt((x[j] - y[j]) ** 2)
        return total

    def predict(self, tests):

        pred = []
        for test in tests:
            distance = np.empty(0)
            # get all distance between test_ & train_s
            for j in range(len(self.x)):
                distance = np.append(distance, self.ocliden(self.x[j], test))

            min_dict = []
            for j in range(self.k):
                index = np.where(distance == distance.min())[0][0]
                min_dict.append(self.y[index])
                # delete min
                distance = np.delete(distance, [index])

            pred.append(np.median(min_dict).astype(int))

        return pred
