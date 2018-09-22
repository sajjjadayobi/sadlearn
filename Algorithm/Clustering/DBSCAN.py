import numpy as np


class DBscan:
    def __init__(self, m, eps=0.5, m_p=5):
        self.m = m
        self.eps = eps
        self.m_p = m_p

    @staticmethod
    def oqlidus(m, p, eps):
        seeds = []
        for i in range(m.shape[0]):
            if np.sqrt(np.power(m[p] - m[i], 2).sum()) < eps:
                seeds.append(i)
        return seeds

    def _expand_cluster(self, labels, i, class_id):
        seeds = self.oqlidus(self.m, i, self.eps)
        if len(seeds) < self.m_p:
            labels[i] = None
            return False
        else:
            for i in seeds:
                labels[i] = class_id

            while len(seeds) > 0:
                current = seeds[0]
                results = self.oqlidus(self.m, current, self.eps)
                if len(results) >= self.m_p:
                    for i in range(len(results)):
                        point = results[i]
                        if labels[point] is None:
                            seeds.append(point)
                            labels[point] = class_id
                seeds = seeds[1:]

            return True

    def labels(self):
        class_id = 1
        labels = np.array([None] * self.m.shape[0])

        for i in range(self.m.shape[0]):
            if not labels[i]:
                if self._expand_cluster(labels, i, class_id):
                    class_id += 1
        return labels
