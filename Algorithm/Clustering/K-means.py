import numpy as np

"""
K-means.py
detail:  Clustering Algorithm 

author: sajjad ayobi
see others in repository : sadlearn
in URL: https://github.com/sajjjadayobi/sadlearn/

date : 10/5/2018
"""


class KMeans:
    """
    K-means is a Algorithm for clustering
        in this algorithm should entering number of clusters


    Parameters
    ----------
    n_cluster : int, optional, default: 2
        The number of clusters to form as well as the number of
        centroids to generate.


    max_iter : int, default: 10
        Maximum number of iterations of the k-means algorithm for a single run.


    Attributes
    ----------
    x: list or np.ndarray
        we data

    clusters : array, [n_clusters, n_features]
        Coordinates of cluster centers

    values: array, [class1,class2,...]
        list that classifies classes

    inertia :  float
        Sum of squared distances of samples to their closest cluster center.

    Examples
    --------
    >>> km = KMeans(list(), n_cluster=3)
    >>> new_values , centers = km.predict()
    """

    def __init__(self, x, n_cluster, max_iter=10):
        self.x = x
        self.max_iter = max_iter
        self.inertia_ = None
        self.clusters = self.random_clusters(x, n_cluster)

    @staticmethod
    def random_clusters(x, n_cluster):
        np.random.seed(42)
        centers = np.array(x[np.random.choice(range(len(x)), size=n_cluster, replace=False)])
        return centers

    @staticmethod
    def manhattan(x, y):
        total = 0
        for j in range(y.shape[0]):
            total += np.abs((x[j] - y[j]))
        return total

    def refresh_clusters(self, x):
        c = np.zeros((len(self.clusters), self.clusters.shape[1]))
        for i, k in enumerate(x):
            m = np.mean(k, axis=0)
            for j in range(len(m)):
                c[i][j] = m[j]
        return c

    def distance(self):
        new = []
        for i in range(len(self.clusters)):
            new.append([])
        for k in range(len(self.x)):
            distance = []
            for i in self.clusters:
                distance.append(self.manhattan(self.x[k], i))

            new[np.argmin(distance)].append(list(self.x[k]))

        return new

    def inertia(self, x):
        distance = []
        for i, v in enumerate(x):
            distance.append(np.sum(np.square(v - self.clusters[i])))
        return np.sum(distance)

    def predict(self):
        for i in range(self.max_iter):
            x = self.distance()
            self.clusters = self.refresh_clusters(x)

        self.inertia_ = self.inertia(x)
        return x
