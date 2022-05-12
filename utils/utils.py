import numpy as np


def xlogx(x):
    ''' x:  nx1 vector '''
    n = len(x)
    out = np.zeros((n, 1))

    for i in range(n):
        if 0 == x[i]:
            out[i] = 0
        else:
            out[i] = x[i] * np.log(x[i])
    return out


"""
if x is positive we are simply using 1 / (1 + np.exp(-x))
but when x is negative we are using the function np.exp(x) / (1 + np.exp(x))
instead of using 1 / (1 + np.exp(-x)) because when x is negative -x will
be positive so np.exp(-x) can explode due to large value of -x.
"""


def sigmoid(x):
    pos_mask = (x >= 0)
    neg_mask = (x < 0)
    z = np.zeros_like(x, dtype=float)
    z[pos_mask] = np.exp(-x[pos_mask])
    z[neg_mask] = np.exp(x[neg_mask])
    top = np.ones_like(x, dtype=float)
    top[neg_mask] = z[neg_mask]
    return top / (1 + z)
