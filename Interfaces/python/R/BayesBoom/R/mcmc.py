import numpy as np


def suggest_burn(loglike, fraction: float = 0.1, quantile: float = 0.9):
    """Suggest a burn-in sample size for an MCMC.  The suggestion uses a heuristic
    based on the time series of log likelihood values by the MCMC sampler.  The
    algorithm looks at the last 'fraction' of the log likelihood sequence and
    finds a specified quantile to use as a threshold.  The burn-in period ends
    with the first element in the chain to meet or exceed this threshold.

    :param log.likelihood:
        MCMC sample path of log likelihood.

    :param fraction:
        The final fraction of the chain available for to determining the log
        likelihood threshold.

    :param quantile:
        The quantile of the values in the final fraction used to determine the
        threshold.

    :return burn: int
        The number of burn-in iterations that should be discarded.  This can be
        0 if 'fraction' is zero, which signals that no burn-in is desired.

    """
    if fraction < 0:
        # A signal that no burnin is desired.
        return 0
    if fraction > 1.0:
        raise Exception("'fraction' must be a number between 0 and 1.")
    cutpoint = int((1.0 - fraction) * len(loglike))
    if cutpoint == 0:
        return 0
    min_loglike = np.quantile(loglike[cutpoint:], quantile)
    burn = np.where(loglike >= min_loglike)[0][0]
    if burn < 0:
        burn = 0
    return int(burn)
