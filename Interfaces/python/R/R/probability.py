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


def rmarkov(n: int, P: np.ndarray, pi0=None):
    """
    Simulation a Markov chain of length n from transition probability matrix P,
    with initial distribution pi0.  The state space is the set of integers in
    {0, ..., S-1} where S is the number of rows in P.

    Args:
      n:  The length of the Markov chain to simulate.
      P: A transition probability matrix.  A square matrix of non-negative
         elements, where each row sums to 1.
      pi0: The distribution of the state at time 0.  A vector of non-negative
        numbers summing to 1.

    Returns:
      A numpy array of integer dtype containing the simulated Markov chain..
    """
    assert(isinstance(P, np.ndarray))
    assert(len(P.shape) == 2)
    S = P.shape[0]
    assert(P.shape[1] == S)
    assert(S > 0)
    if pi0 is None:
        pi0 = np.ones(S) / S
    assert(len(pi0) == S)

    ans = np.full(n, -1)
    ans[0] = np.random.choice(range(S), p=pi0)
    for i in range(1, n):
        ans[i] = np.random.choice(range(S), p=P[ans[i-1], :])
    return(ans)
