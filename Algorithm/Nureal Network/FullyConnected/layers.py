import numpy as np


def liner_forward(x, w, b):
    return x @ w + b


def liner_backward(dout, x, w):
    dw = x.T @ dout
    dx = dout @ w.T
    db = np.sum(dout, axis=0)
    return dx, dw, db


def relu(z):
    return np.maximum(0, z)


def max_backward(dout, local):
    # relu backward
    return dout * (local > 0)


def dropout_forward(x, kp):
    prob = 1 - kp
    # for accuracy on x_test : Ansamble learning
    # mean of all 2^noeron_size newtwork
    mask = (np.random.rand(*x.shape) < prob) / prob
    x_dorp = x * mask
    return x_dorp, mask


def dropout_backward(dout, mask):
    return dout * mask


def batchnorm_forward(x, gamma, beta, test_param, mode='train'):
    running_mean = test_param.setdefault('run_mean', np.zeros(x.shape[1], ))
    running_var = test_param.setdefault('run_var', np.zeros(x.shape[1], ))
    cache = None

    if mode == 'train':
        # Normalize
        mu = np.mean(x, axis=0)
        xc = x - mu
        var = np.mean(xc ** 2, axis=0)
        std = np.sqrt(var + 1e-5)
        x_norm = xc / std

        # Scale and Shift
        out = gamma * x_norm + beta
        cache = (x, xc, var, std, x_norm, gamma)

        # update running mean and running average
        test_param['run_mean'] = .9 * running_mean + (1 - .9) * mu
        test_param['run_var'] = .9 * running_var + (1 - .9) * var
    else:
        x_norm = (x - test_param['run_mean']) / (np.sqrt(test_param['run_var'] + 1e-5))
        out = gamma * x_norm + beta

    return out, cache


def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.
    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.
    Returns:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
     """
    x, xc, var, std, xn, gamma = cache
    N = x.shape[0]

    dbeta = np.sum(dout, axis=0)
    dgamma = np.sum(dout * xn, axis=0)
    dxn = dout * gamma

    dxc = dxn / std
    dstd = np.sum(-(xc * dxn) / (std * std), axis=0)
    dvar = 0.5 * dstd / std

    dxc += (2.0 / N) * xc * dvar
    dmu = -np.sum(dxc, axis=0)
    dx = dxc + dmu / N

    return dx, dgamma, dbeta


def mini_batch(x, y, size=64):
    # mini batch gradient
    indexs = np.random.choice(x.shape[0], size, replace=False)
    return x[indexs], y[indexs]




def softmax_loss(scores, y, param, reg=0):
    # forward step: computing data loss
    num = scores.shape[0]
    scores -= np.max(scores, axis=1, keepdims=True)
    exp = np.exp(scores)
    probs = exp / np.sum(exp, axis=1, keepdims=True)
    loss = -np.log(probs[range(num), y])
    loss = np.mean(loss)

    if reg != 0:
        w_loss = 0
        for i in range(3):
            w_loss += np.sum(param['w%d' % i] ** 2)
        loss += .5 * reg * w_loss

        # backward step: computing Dervative with Respect to output
    dscore = probs
    dscore[np.arange(num), y] -= 1
    dscore /= num
    return loss, dscore