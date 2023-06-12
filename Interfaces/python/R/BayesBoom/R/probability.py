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


def rbeta(n, a=1, b=1):
    from scipy.stats import beta
    return beta.rvs(a, b, size=n)


def rpois(n, lam):
    from scipy.stats import poisson
    return poisson.rvs(lam, size=n)


def rbinom(n, size, prob):
    from scipy.stats import binom
    return binom.rvs(size, prob, size=n)


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


def dmvn(y, mu, Sigma, inv=False, logscale=False):
    """
    Multivariate normal density.

    Args:
      y: Either a vector or a 2-D array.  If a vector is passed then the output
        is a scalar density value.  If a matrix is passed, the return value is
        a vector with each element giving the density for the corresponding row
        of y.
      mu: If y is a vector, mu must be a vector of the same dimension.  If y is
       a matrix, then mu is either a vector with lenght matching the number of
       columns in y, or a matrix with the same shape as y.
      Sigma: If y is a vector Sigma is a symmetric positive definite matrix
        matching the length of y.  If y is a matrix then either Sigma is a
        single matrix matching the length of y, or a 3-way array with first
        dimension matching the first dimension of y, and Sigma[i, :, :] giving
        the variance of y[i, :].
      inv: If True then Sigma represents the precision of y (the inverse
        variance).  If False then Sigma represents the variance.
      logscale: If True then the log of the density is returned.  If False then
        the regular density value is returned.
    """

    y = np.array(y)
    if len(y.shape) > 2:
        raise Exception("dmvn requires either a vector or matrix input for y.")
    elif len(y.shape) == 1:
        y = y.reshape((1, -1))
    nobs = y.shape[0]
    dim = y.shape[1]

    mu = np.array(mu)
    if len(mu.shape) > 2:
        raise Exception("dmvn requires either a vector or matrix input for mu.")
    if len(mu.shape) == 1:
        mu = mu.reshape((1, -1))
    if mu.shape[0] == 1 and nobs > 1:
        mu = np.array([mu.ravel()] * nobs)
    if mu.shape[0] != nobs:
        raise Exception("The shapes of mu and y must match.")

    if len(Sigma.shape) > 4:
        raise Exception(
            "dmvn requires either a matrix or 3-way array for Sigma")
    if len(Sigma.shape) < 2:
        raise Exception("Sigma can't be a vector in dmvn.")
    if len(Sigma.shape) == 2:
        if not np.allclose(Sigma, Sigma.T):
            raise Exception(
                "Sigma must be a symmetric positive definite matrix.")
        if not inv:
            Sigma = np.linalg.inv(Sigma)
            inv = True
        ldsi = np.array([np.linalg.slogdet(Sigma)[1]] * nobs)
        Sigma = np.array([Sigma] * nobs)
    if len(Sigma.shape) != 3:
        raise Exception("Something with wrong with Sigma")
    if Sigma.shape[0] == 1 and nobs > 1:
        if not inv:
            Sigma = np.linalg.inverse(Sigma[0, :, :])
            inv = True
        else:
            Sigma = Sigma[0, :, :]
        ldsi = np.array([np.linalg.slogdet(Sigma)] * nobs)[1]
        Sigma = np.array([Sigma] * nobs)
    if Sigma.shape[0] != nobs or Sigma.shape[1] != dim or Sigma.shape[2] != dim:
        raise Exception("The shapes of y and Sigma must match.")
    if not inv:
        Sigma = np.linalg.inv(Sigma)
        ldsi = np.linalg.slogdet(Sigma)[1]

    log2pi = 1.83787706641
    residual = y - mu

    qform = np.einsum('ij,ijk,ik -> i', residual, Sigma, residual)

    ans = 0.5 * (-dim * log2pi + ldsi - qform)
    if nobs == 1:
        ans = float(ans)
    if logscale:
        return ans
    return np.exp(ans)


def rmvn(n, mu, Sigma, drop=True):
    """
    Draws from the multivariate normal distribution with mean mu and variance
    matrix Sigma.

    Args:

      n: The number of desired draws.

      mu: A numpy vector or matrix giving the mean of the draws.  If a vector
        then mu is the mean for all .  If a matrix then mu[i, :] is the mean
        vector for draw i.

      Sigma: Either a 2D or a 3D numpy array.  If a 2D array is passed then
        Sigma is the common variance matrix used for all draws.  If a 3D array
        is passed, then Sigma[i, :, :] is the variance matrix for draw i.
    """
    mu = np.array(mu)
    if len(mu.shape) > 2:
        raise Exception("dmvn requires either a vector or matrix input for mu.")
    if len(mu.shape) == 1:
        mu = mu.reshape((1, -1))
    if mu.shape[0] == 1 and n > 1:
        mu = np.array([mu.ravel()] * n)
    if mu.shape[0] != n:
        raise Exception(f"Requested {n} draws but passed a 'mu' argument with"
                        f" {mu.shape[0]} rows.")

    dim = mu.shape[1]
    Z = np.random.randn(n, dim)

    if len(Sigma.shape) == 2:
        L = np.linalg.cholesky(Sigma)
        draws = (L @ Z.T).T + mu
    elif len(Sigma.shape) != 3:
        raise Exception(
            "Either a matrix or a 3-way array is required for Sigma")
    else:
        draws = np.array([
            np.linalg.cholesky(Sigma[i, :, :]) @ Z[i, :] + mu[i, :]
            for i in range(n)])

    if n == 1 and drop:
        return draws[0, :]
    else:
        return draws


def rmulti(probs, n=1):
    """
    Simulate one or more draws from the given discrete probability
    distribution.
    """
    return np.random.choice(range(len(probs)), size=n, replace=True, p=probs)
