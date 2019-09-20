import scipy.stats as sp
import numpy as np
from numbers import Number

# expose R's equivalent of pnorm, qnorm, etc.


def as_numeric(x):
    return np.array(x)


def dnorm(x, mu=0, sigma=1, log=False):
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


def pnorm(x, mu=0, sigma=1, lower=True, log=False):
    """
    Normal cumulative distribuion function.

    """
    if log:
        if lower:
            return sp.norm.logcdf(x, loc=mu, scale=sigma)
        else:
            return sp.norm.logsf(x, loc=mu, scale=sigma)
    else:
        if lower:
            return sp.norm.cdf(x, loc=mu, scale=sigma)
        else:
            return sp.norm.sf(x, loc=mu, scale=sigma)


def qnorm(x, mu=0, sigma=1, lower=True, log=False):
    """
    Quantiles of the normal distribuion.

    """
    if log:
        x = np.exp(x)

    if lower:
        return sp.norm.ppf(x, mu, sigma)
    else:
        return sp.norm.ppf(1 - x, mu, sigma)


def rnorm(n, mu=0, sigma=1):
    """
    Random deviates from the normal distribution.

    """

    return np.random.randn(n) * sigma + mu
