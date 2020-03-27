import scipy.stats as ss


def density(x):
    """Returns a kernel density estimate of x.  Suitable for use with
    lines_gaussian_kde in plots.py.

    """
    return ss.gaussian_kde(x)
