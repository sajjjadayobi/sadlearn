import numpy as np
import matplotlib.pyplot as plt


def sum_square_error(theta, x, y):
    slop = []
    for i, t in enumerate(theta[1:]):
        slop.append(x[:, i] * t)
    y_pred = (np.sum(slop, axis=0) + theta[0]).reshape(-1, 1)

    ssr = np.sum(np.square(y_pred - y)) / len(x)
    return ssr, y_pred


def gradient_descent(theta, x, y, alpha):
    x0 = 1
    new_theta = np.empty(0)
    for i in range(len(theta)):
        gradient = np.sum(- (2 / len(y)) * x0 * (y - ((theta[1:] * x) + theta[0])))
        new_theta = np.append(new_theta, gradient * alpha)
        x0 = x

    theta -= new_theta
    return theta


def fit_best_line(x, y, alpha):
    theta = [1 for _ in range(x.shape[1] + 1)]

    error = np.inf
    for i in range(1000000):
        theta = gradient_descent(theta, x, y, alpha)
        new_error, y_pred = sum_square_error(theta, x, y)
        if new_error + 0.00009 >= error:
            print("finish iterator :", i)
            break
        error = new_error

    return y_pred, error


X = np.arange(0, 10).reshape(-1, 1)
Y = np.array([17, 18, 20, 19, 20, 21, 20, 19, 18, 20]).reshape(-1, 1)

Y_pred, Error = fit_best_line(X, Y, 0.033)

plt.scatter(X[:, 0], Y)
plt.plot(Y_pred[:, 0], c='r')
plt.show()
