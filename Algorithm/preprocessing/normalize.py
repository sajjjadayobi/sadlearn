import numpy as np

"""
normalization.py

author: sajjad ayobi
see others in repository : sadlearn
in URL: https://github.com/sajjjadayobi/sadlearn/

date : 10/4/2018
"""


def min_max_normalize(x, min=0, max=1):
    """
    Returns
    -------
    normalization data in favorite range

    Parameters
    ----------
    x: x is we data in shape Matrix

    min: minimum of data normalized
    max: maximum of data normalized

    Examples
    --------
    >>> import numpy as np
    >>> data = np.array([[1, 2, 3],
    >>>                  [4, 5, 6],
    >>>                  [7, 8, 9]])

    >>> norm = min_max_normalize(data, 0, 1)
    >>> print(norm)

    >>>     [[0.    0.125 0.25 ]
    >>>     [0.375 0.5   0.625]
    >>>     [0.75  0.875 1.   ]]
    """

    a = max - min
    b = np.max(x) - np.min(x)
    c = x - np.max(x)
    return ((a / b) * c) + max

def l1_norm(x):
    """
    Returns
    -------
    normalization of data in range(0,1) with method manhattan distance

    Parameters
    ----------
    x: x is we data in shape Matrix

    note
    --------
    most import numpy as np
    """
    a = x - np.min(x)
    b = np.max(x) - np.min(x)
    return a / b


def l2_norm(x):
    """
    Returns
    -------
    normalization of data in range(0,1) with method euclidean distance

    Parameters
    ----------
    x: x is we data in shape Matrix

    note
    --------
    most import numpy as np
    """
    a = np.sqrt(np.square(x))
    size_of_vector = np.sqrt(np.sum(a ** 2, axis=1)).reshape(len(x), 1)

    return x / size_of_vector



