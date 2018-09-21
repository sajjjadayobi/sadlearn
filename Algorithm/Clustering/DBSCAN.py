import numpy as np

class DBscan:

    def __init__(self, m, eps=0.5, m_p=5):
        self.NOISE = None
        self.m = m
        self.eps = eps
        self.m_p = m_p

    @staticmethod
    def oqlidus(m, p, eps):
        seeds = []
        for i in range(m.shape[1]):
            if np.sqrt(np.power(m[:, p] - m[:, i], 2).sum()) < eps:
                seeds.append(i)
        return seeds

    def _expand_cluster(self, m, labels, i, class_id):
        seeds = self.oqlidus(m, i, self.eps)
        if len(seeds) < self.m_p:
            labels[i] = self.NOISE
            return False
        else:
            for i in seeds:
                labels[i] = class_id

            while len(seeds) > 0:
                current_point = seeds[0]
                results = self.oqlidus(m, current_point, self.eps)
                if len(results) >= self.m_p:
                    for i in range(len(results)):
                        result_point = results[i]
                        if not labels[result_point]:
                            seeds.append(result_point)
                            labels[result_point] = class_id

                seeds = seeds[1:]
            return True

    def labels(self):

        class_id = 1
        n_points = self.m.shape[1]
        labels = [False] * n_points

        for i in range(n_points):
            if not labels[i]:
                if self._expand_cluster(self.m, labels, i, class_id):
                    class_id += 1

        return labels
