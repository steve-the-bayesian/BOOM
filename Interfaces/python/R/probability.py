import scipy.stats as sp
import numpy as np
from numbers import Number

# expose R's equivalent of pnorm, qnorm, etc.


def as_numeric(x):
    return np.array(x)


def dnorm(x, mu, sigma, log=False):
    """Normal density function.

    :param x: number, list or numpy array
        The argument of the density function.

    :param x: number, list or numpy array
        The argument of the density function.

    """
    x = as_numeric(x)
    if log:
        return sp.norm.logpdf(x, mu, sigma)
    else:
        return sp.norm.pdf(x, mu, sigma)

def pnorm(x, mu, sigma, log=False):
    """
    """
