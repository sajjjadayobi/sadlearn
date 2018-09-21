import numpy as np


def computing_m(x, y):
    x_mean = (x - np.mean(x))
    y_mean = (y - np.mean(y))
    x_m_sq = np.square(x_mean).sum()

    m = (x_mean * y_mean).sum() / x_m_sq
    return m


def linReg(x, y):
    m = computing_m(x, y)
    c = np.mean(y) - m * np.mean(x)

    reg = np.empty(0)
    for i in x:
        reg = np.append(reg, m * i + c)

    return reg.reshape(-1, 1)


x = np.arange(0, 10)
y = np.array([17, 18, 20, 19, 20, 21, 20, 19, 18, 20])

reg = linReg(x, y)
print(reg)
