import numpy as np

"""
KNN.py
detail:  Clustering Algorithm 

author: sajjad ayobi
see others in repository : sadlearn
in URL: https://github.com/sajjjadayobi/sadlearn/

date : 10/6/2018
"""


def ocliden(x, y):
    """
     ocliden distance :
     calculate distance between tow samples
    """
    total = 0
    for j in range(len(y)):
        total += np.sqrt((x[j] - y[j]) ** 2)
    return total


def manhattan(x, y):
    """
     manhattan distance :
     calculate distance between tow samples
    """
    total = 0
    for j in range(len(y)):
        total += np.abs((x[j] - y[j]))
    return total


def minkowski(x, y, p):
    """
     minkowski distance :
     calculate distance between tow samples
    """
    total = 0
    for j in range(len(y)):
        total += (np.abs((x[j] - y[j]) ** p).sum(axis=0)) ** (1 / p)
    return total


def knn(train_, test_, target_, k, method=ocliden):
    """
    K-Nearest neighbors: the algorithm for classification with use Nearest neighbors
    https://en.wikipedia.org/wiki/K-nearest_neighbor_algorithm

    Parameters
    ----------
    :param train_: train data for learning
    :param test_: test data for validation and predict
    :param target_: correct response of train data
    :param k: number of neighbors
    :param method: distance validations method

    :return: predict values for test data

    Examples
    --------
    >>> predict = knn(train_set, test_set, targets, 3)
    >>> print(predict)
    """
    predict = []
    for test in test_:
        space = np.empty(0)
        # getting all distance between test_ & train_
        for j in range(len(train_)):
            space = np.append(space, method(train_[j], test))

        result = []
        for j in range(k):
            index = np.where(space == space.min())[0][0]
            result.append(target_[index])
            # delete minimum
            space = np.delete(space, [index])

        predict.append(np.median(result).astype(int))

    return predict
