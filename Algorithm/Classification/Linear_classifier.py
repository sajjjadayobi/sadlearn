import numpy as np


class liner_classifier:
    """
    Liner Classifier with Gradient descent Algorithm
    and find best Weight for test data

    note: better use Normalized data

    Parameter
    ---------
    batch_size: int
    size of batch in Gradient descent

    n_iter: int
    number of iteration Gradient Descent

    learning_rate: float
    alpha in Gradient Descent Algorithm

    report: bool
    showing report of each iter in Gradient Descent

    Return
    ------
    predicted label for data

    Examples
    --------
    >>> clf = liner_classifier()
    >>> clf.fit(x, y)
    >>> y_pred = clf.predict(x_test)
    """

    def __init__(self, batch_size, n_iter=1000, learning_rate=0.00001, report=True):
        self.weight = None
        self.bias = None
        self.batch_size = batch_size
        self.lr = learning_rate
        self.n_iter = n_iter
        self.report = report

    # Loss function for multi classifier
    @staticmethod
    def softmax_loss(scores, y):
        # forward step: computing data loss
        num = scores.shape[0]
        scores -= np.max(scores, axis=1, keepdims=True)
        exp = np.exp(scores)
        probs = exp / np.sum(exp, axis=1, keepdims=True)
        loss = -np.log(probs[range(num), y])
        loss = np.mean(loss)
        # backward step: computing Derivative with Respect to output
        d_out = probs
        d_out[np.arange(num), y] -= 1
        return loss, d_out

    # computing scores of each class
    @staticmethod
    def affine_forward(x, w, b):
        return x @ w.T + b

    # computing Derivative with Respect to input
    @staticmethod
    def affine_backward(x, d_out):
        d_weight = np.dot(x.T, d_out)
        d_bais = np.sum(d_out, axis=0)
        return d_weight, d_bais

    def fit(self, x, y):
        # random Init weight
        w = np.random.randn(len(set(y)), x.shape[1]) * 0.001
        b = np.zeros((len(set(y)),))

        report_time = int(self.n_iter / 10)
        for i in range(self.n_iter + 1):
            index = np.random.choice(x.shape[0], self.batch_size, replace=False)
            x_batch = x[index]
            y_batch = y[index]

            # Loss
            scores = self.affine_forward(x_batch, w, b)
            loss, dout = self.softmax_loss(scores, y_batch)
            # Optimize Weight
            dw, db = self.affine_backward(x_batch, dout)
            w -= self.lr * dw.T
            b -= self.lr * db

            # report
            if self.report and i % report_time == 0:
                y_pred = np.argmax(scores, axis=1)
                score = np.mean(y_pred == y_batch)
                scores = self.affine_forward(x, w, b)
                y_pred = np.argmax(scores, axis=1)
                train = np.mean(y_pred == y)
                print('\t  iter %4d loss= %2.5f | batch= %0.2f | all= %0.3f' % (i, loss, score, train))

        self.weight = w
        self.bias = b

    def predict(self, x):
        scores = self.affine_forward(x, self.weight, self.bias)
        return np.argmax(scores, axis=1)
