import numpy as np
from scipy.stats import norm
from Algorithm.preprocessing.normalize import l2_norm


class baysian:

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.labels = set(self.y)

        self.std_label = np.zeros((len(self.labels), x.shape[1]))
        self.mean_label = self.std_label
        self.prior = [self.x.shape[1] / len(self.y)] * len(self.labels)

    def fit(self):

        for i in self.labels:
            label = self.x[np.where(self.y == i)]
            self.std_label[i] = np.std(label, axis=0)
            self.mean_label[i] = np.mean(label, axis=0)

        return self.prior

    def predict(self, x_test):

        pred = np.empty(0)
        for obs in x_test:
            evidence = l2_norm(x_test)
            evidence = np.prod(evidence)

            likelihood = norm.pdf((obs - self.mean_label) / self.std_label)
            likelihood = np.prod(likelihood, axis=1)

            posterior = self.prior * likelihood / (evidence + 0.001)
            pred = np.append(pred, np.argmax(posterior))

        return pred.astype(int)

    def score(self, x_test, y_test):
        y_pred = self.predict(x_test)
        score = np.mean(y_pred == y_test)
        return score
