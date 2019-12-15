from scipy.stats import norm

class NaiveBayesian:
    
    def __init__(self, x, y, class_names=None):
        self.x = x
        self.y = y
        self.class_names = class_names
        self.labels = set(self.y)

        self.std_label = np.zeros((len(self.labels), x.shape[1]))
        self.mean_label = self.std_label
        self.prior = [self.x.shape[1] / len(self.y)] * len(self.labels)

    
    @staticmethod
    def l2_norm(x):
        a = np.sqrt(np.square(x))
        size_of_vector = np.sqrt(np.sum(a ** 2, axis=1)).reshape(len(x), 1)
        return x / size_of_vector
    
    def fit(self):
        for i in self.labels:
            label = self.x[np.where(self.y == i)]
            self.std_label[i] = np.std(label, axis=0)
            self.mean_label[i] = np.mean(label, axis=0)

        return self.prior


    def predict(self, x):
        if len(x.shape)==1:
            x = x[np.newaxis, :]
            
        # Maximum Posterior
        evidence = np.prod(self.l2_norm(x)) + 0.001
        p = np.empty(0)
        for obs in x:
            likelihood = norm.pdf((obs - self.mean_label) / self.std_label)
            likelihood = np.prod(likelihood, axis=1)

            posterior = self.prior * likelihood / evidence
            p = np.append(p, np.argmax(posterior))

        if isinstance(self.class_names, np.ndarray):
            return self.class_names[p.astype(int)]
        return p


    def score(self, x_test, y_test):
        y_pred = self.predict(x_test)
        score = np.mean(y_pred == y_test)
        return score
