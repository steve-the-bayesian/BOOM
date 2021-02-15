import scipy.stats as ss
import numpy as np


def sd(x):
    """
    Return the standard deviation of x.  If x is a matrix then return standard
    deviation of its columns.
    """
    if hasattr(x, "shape") and len(x.shape) == 2:
        return np.std(x, ddof=1, axis=0)
    else:
        return np.std(x, ddof=1)


def density(x):
    """Returns a kernel density estimate of x.  Suitable for use with
    lines_gaussian_kde in plots.py.

    """
    return ss.gaussian_kde(x)
