import numpy as np


def SGD(w, dw, lr, *wargs):
	# Stochastic Gradient descent
    w -= dw * lr
    return w, None
	
def Rmsprop(x, dx, lr, config):
    """ 
    Uses the RMSProp update rule, which uses a moving average of squared
    gradient values to set adaptive per-parameter learning rates.
    config format:
    - decay_rate: Scalar between 0 and 1 giving the decay rate for the squared
    - cache: Moving average of second moments of gradients.
    """  
    if config is 0: config = {}
    config.setdefault('cache', np.zeros_like(x))
    cache = config['cache']
    decay_rate = 0.9

    cache = decay_rate * cache + (1 - decay_rate) * dx**2
    next_x = x - lr * dx / (np.sqrt(cache) + 1e-8)

    config['cache'] = cache
    return next_x, config
	
	
	
def Momentum(w, dw, lr, config):
    """
    Performs stochastic gradient descent with momentum.
    config format:
    - momentum: Scalar between 0 and 1 giving the momentum value.
    """
    if config is 0: config = {}
    config.setdefault('v', np.zeros_like(w))

    v = .9 * config['v'] - lr * dw
    w += v
    
    config['v'] = v
    return w, config
	
	
	
def Nestrov_momentum(w, dw, lr, config):
    """ or nag
    Performs stochastic gradient descent with momentum.
    config format:
    - momentum: Scalar between 0 and 1 giving the momentum value.
    """
    if config is 0: config = {}
    config.setdefault('v', np.zeros_like(w))
    mu = .9
    v_pred = config['v']
    
    v = mu * config['v'] - lr * dw
    w += -mu * v_pred + (1+mu) * v
    config['v'] = v

    return w, config
	


def Adam(x, dx, lr, config):
    """ config format:
    - beta1: Decay rate for moving average of first moment of gradient.
    - beta2: Decay rate for moving average of second moment of gradient.
    - m: Moving average of gradient.
    - v: Moving average of squared gradient.
    - t: Iteration number. """
    
    if config is 0: config={}
    config.setdefault('t', 1)
    config.setdefault('m', np.zeros_like(x))
    config.setdefault('v', np.zeros_like(x))

    # read params from dictionary
    beta1, beta2 = .9, .999
    m, v, t = config['m'], config['v'], config['t']

    t += 1
    # apply adam update rule
    m =  beta1 * m + (1 - beta1) * dx
    v =  beta2 * v + (1 - beta2) * dx ** 2
    mb = m / (1 - beta1 ** t)
    vb = v / (1 - beta2 ** t)
    next_x = x - lr * mb / (np.sqrt(vb) + 1e-8)

    # store new params in the dictionary
    config['m'], config['v'] = m, v
    return next_x, config