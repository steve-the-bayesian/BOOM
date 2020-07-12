import scipy.stats as sp
import numpy as np

# expose R's equivalent of pnorm, qnorm, etc.


def as_numeric(x):
    return np.array(x)


def dnorm(x, mean=0, sd=1, log=False):
    """Normal density function.

    :param x: number, list or numpy array
        The argument of the density function.

    :param x: number, list or numpy array
        The argument of the density function.

    """
    x = as_numeric(x)
    if log:
        return sp.norm.logpdf(x, mean, sd)
    else:
        return sp.norm.pdf(x, mean, sd)


def pnorm(x, mean=0, sd=1, lower=True, log=False):
    """
    Normal cumulative distribuion function.

    """
    if log:
        if lower:
            return sp.norm.logcdf(x, loc=mean, scale=sd)
        else:
            return sp.norm.logsf(x, loc=mean, scale=sd)
    else:
        if lower:
            return sp.norm.cdf(x, loc=mean, scale=sd)
        else:
            return sp.norm.sf(x, loc=mean, scale=sd)


def qnorm(x, mean=0, sd=1, lower=True, log=False):
    """
    Quantiles of the normal distribuion.

    """
    if log:
        x = np.exp(x)

    if lower:
        return sp.norm.ppf(x, mean, sd)
    else:
        return sp.norm.ppf(1 - x, mean, sd)


def rnorm(n, mean=0, sd=1):
    """
    Random deviates from the normal distribution.

    """

    return np.random.randn(n) * sd + mean


def dgamma(y, shape, scale, log=False):
    from scipy.stats import gamma
    if log:
        return gamma.logpdf(y, shape, scale=1.0 / scale)
    else:
        return gamma.pdf(y, shape, scale=1.0 / scale)


def pgamma(y, shape, scale, lower_tail=True, log=False):
    from scipy.stats import gamma
    if lower_tail:
        if log:
            return gamma.logcdf(y, shape, scale=1.0 / scale)
        else:
            return gamma.cdf(y, shape, scale=1.0 / scale)
    else:
        if log:
            return gamma.logsf(y, shape, scale=1.0 / scale)
        else:
            return gamma.sf(y, shape, scale=1.0 / scale)


def qgamma(probs, shape, scale, lower_tail=True, log=False):
    from scipy.stats import gamma
    if log:
        probs = np.exp(probs)
    if lower_tail:
        return gamma.ppf(probs, shape, scale=1.0 / scale)
    else:
        return gamma.isf(probs, shape, scale=1.0 / scale)


def rgamma(n, shape, scale):
    """
    Random deviates from the gamma distribution with mean shape / scale.
    """
    from scipy.stats import gamma
    return gamma.rvs(shape, size=n, scale=1.0 / scale)


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
    P = np.array(P)
    if len(P.shape) != 2 or P.shape[0] != P.shape[1]:
        raise Exception("P must be a square matrix.")
    S = P.shape[0]
    if pi0 is None:
        pi0 = np.ones(S) / S
    else:
        pi0 = np.array(pi0).flatten()

    if len(pi0) != S:
        raise Exception("Initial distribution must have the same "
                        "dimension as the transition probability matrix.")
    ans = np.full(n, -1)
    ans[0] = np.random.choice(range(S), p=pi0)
    for i in range(1, n):
        ans[i] = np.random.choice(range(S), p=P[ans[i-1], :])
    return(ans)
