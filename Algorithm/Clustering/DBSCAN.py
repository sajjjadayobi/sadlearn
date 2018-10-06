import numpy as np

"""
DBSCAN.py
detail:  Clustering Algorithm 

author: sajjad ayobi
see others in repository : sadlearn
in URL: https://github.com/sajjjadayobi/sadlearn/

date : 10/5/2018
"""


class DBSCAN:
    """
    DBSCAN is a Algorithm for clustering also can find Noise in data
    Good for data which contains clusters of similar density

    Parameters
    ----------
    x: list or np.ndarrary , most
        Data in the from of a matrix

    epsilon: float, optional
        maximum distance between tow sample in one class

    min_point: int, optional
        minimum number of sample in one class
        if neighbor of a sample was less than min_point, this sample is a Noise

    Attributes
    ----------
    labels: list of class number

    Note
    ----
    -1 in labels means it sample is Noise

    for example
    -----------
    >>> import numpy as np

    >>> db = DBSCAN(np.array())
    >>> labels = db.labels

    """
    def __init__(self, x, epsilon=0.5, min_point=5):
        self.x = x
        self.eps = epsilon
        self.m_p = min_point

    def neighbor_points(self, x, k):
        neighbors = []
        for i in range(len(x)):
            # ouclidean distance
            if np.sqrt(np.square(x[k] - x[i]).sum()) < self.eps:
                neighbors.append(i)
        return neighbors

    def expand_cluster(self, labels, i, class_id):

        neighbors = self.neighbor_points(self.x, i)
        if len(neighbors) < self.m_p:
            labels[i] = -1
            return False

        for i in neighbors:
            labels[i] = class_id
        while len(neighbors) > 0:
            current = neighbors[0]
            new_neighbors = self.neighbor_points(self.x, current)

            if len(new_neighbors) >= self.m_p:
                for i in range(len(new_neighbors)):
                    point = new_neighbors[i]
                    if labels[point] is None:
                        labels[point] = class_id
                        neighbors.append(point)
            del neighbors[0]

        return True

    @property
    def labels(self):
        x = self.x
        labels = np.array([None] * len(x))

        class_id = 1
        for i in range(len(x)):
            if labels[i] is None:
                if self.expand_cluster(labels, i, class_id):
                    class_id += 1
        return labels
