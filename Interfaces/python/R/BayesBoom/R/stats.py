import scipy.stats as ss
import scipy.sparse

import numpy as np
import pandas as pd


def mean(x, axis=None, na_rm=True):
    """
    Return the mean of x, for a variety of types.  Supported types include
    - numpy arrays
    - pandas data frames
    - pandas Series
    - scipy.sparse.matrix
    - scipy.sparse.array

    Args:
      x:  The matrix-like or
    """

    # Support numpy matrix objects -- if they exist in the current version of
    # numpy.
    if "matrix" in dir(np) and isinstance(x, np.matrix):
        x = np.array(x)

    if isinstance(x, pd.DataFrame):
        return x.mean(axis=axis, skipna=na_rm, numeric_only=True)

    elif isinstance(x, pd.Series):
        return x.mean(skipna=na_rm)

    elif isinstance(x, np.ndarray):
        if na_rm:
            return np.nanmean(x, axis=axis)
        else:
            return np.mean(x, axis=axis)

    elif isinstance(x, scipy.sparse.spmatrix):
        if np.min(x.shape) == 0:
            return 0
        ans = x.mean(axis=axis)
        if isinstance(ans, np.matrix):
            ans = np.array(ans)
        if isinstance(ans, np.ndarray):
            ans = ans.ravel()
        return ans


def sd(x, axis=None, na_rm=True):
    """
    Return the standard deviation of x.  If x is a matrix then return standard
    deviation of its columns.  Supported types include
    - np.ndarray
    - pd.DataFrame
    - pd.Series
    - scipy.sparse.spmatrix

    Args:
      x:  The object whose standard deviation is desired.
      axis: The dimension over which to measure the standard deviation.  If x is
        one-dimensional or if x is a pd.DataFrame then this argument is ignored.
      na_rm: Should nan's be omitted from the calculation?  This argument is
        ignored if x is a sparse matrix.

    Returns:
      If x is 2 dimensional, then the return value is a vector containing the
      standard deviations of its columns.  If x is a pd.DataFrame then the
      return value is a pd.Series indexed by the columns of x.  Otherwise the
      return value is a 1-D numpy array.

      If x is 1 dimensional then the return value is a scalar.
    """
    if isinstance(x, scipy.sparse.spmatrix):
        return SparseSd(x, axis=axis)

    if isinstance(x, pd.DataFrame):
        return x.std(ddof=1, skipna=na_rm)

    if "matrix" in dir(np) and isinstance(x, np.matrix):
        x = np.array(x)
        if np.min(x.shape) <= 1:
            x = x.ravel()

    if hasattr(x, "shape") and len(x.shape) == 2 and x.shape[1] > 1:
        # Handle the case where x looks like a "matrix"
        if x.shape[0] <= 1.0:
            return np.zeros(x.shape[1])
        if na_rm:
            return np.nanstd(x, ddof=1, axis=axis)
        else:
            return np.std(x, ddof=1, axis=axis)
    else:
        # Handle the case where x looks like a "vector."
        x = np.array(x).ravel()
        if len(x) <= 1:
            return 0
        if na_rm:
            return np.nanstd(x, ddof=1)
        else:
            return np.std(x, ddof=1)


def quantile(x, probs=np.linspace(0, 1, 5), **kwargs):
    """
    Return the requested quantiles.
    """
    x = pd.DataFrame(x)
    return x.quantile(q=probs, **kwargs)


def SparseSd(x, axis=None):
    """
    Args:
      x: A scipy.sparse matrix.  I.e. an object inheriting from
        scipy.sparse.spmatrix.
      axis:  The axis over which to take the standard deviation.

    Returns:
      If x is a multi column matrix then the return is a numpy array containing
      the standard deviations of the columns in x.  If x is a single column (or
      single row) matrix then the return value is a scalar giving the standard
      deviation of all the elements in x.
    """
    xsq = x.copy()
    xsq.data **= 2

    if np.min(x.shape) == 0:
        return 0

    if np.min(x.shape) == 1:
        sample_size = np.max(x.shape)
        if sample_size <= 1:
            return 0.0
        x_mean_square = xsq.mean()
        x_mean = x.mean()
        variance = x_mean_square - x_mean ** 2
        correction = sample_size / (sample_size - 1)
        return np.sqrt(variance * correction)

    else:
        # Without type coercion, the following objects would be a numpy.matrix,
        # which is an antiquated type.  Use of ravel() prevents an unwanted
        # extra dimension.
        sample_size = x.shape[0]
        if sample_size < 2:
            return np.zeros(x.shape[1])
        x_mean_square = np.array(xsq.mean(axis=axis)).ravel()
        x_mean = np.array(x.mean(axis=axis)).ravel()
        variance = x_mean_square - x_mean ** 2
        correction = sample_size / (sample_size - 1)
        return np.sqrt(variance * correction)


def density(x):
    """Returns a kernel density estimate of x.  Suitable for use with
    lines_gaussian_kde in plots.py.

    """
    return ss.gaussian_kde(x)


def acf(x, lags=40, ax=None, plot=True, correlation=True):
    """
    Compute (and optionally plot) the autocorrelation function (or
    autocovariance function) of a variable x.

    Args:
      x:  A sequence of numbers whose autocorrelation is desired.
      lags:  The number of lags to compute in the ACF.
      ax: A matplotlib.Axes object on which to draw the ACF.  If None and plot
        is True then a new Axes object is created.
      plot:  Should the results be plotted?
      correlation: If True then the autocorrelation function is compuated.
        Otherwise the autocovariance function is computed.

    Returns:
      The autocorrelation or autocovariance function, as a 1-D numpy array.
    """

    import BayesBoom.boom as boom
    from .boom_py_utils import to_boom_vector
    if lags > len(x):
        lags = len(x) - 1

    boom_acf = boom.acf(to_boom_vector(x), lags, correlation).to_numpy()
    if plot:
        if ax is None:
            import matplotlib.pyplot as plt
            _, ax = plt.subplots(1, 1)
        lags = np.arange(len(boom_acf))
        ax.vlines(lags, ymin=0, ymax=boom_acf)

    return boom_acf


def kl_divergence(p1, p2):
    """
    Args:
      p1, p2: Discrete probability distributions.  Both distributions must be
        the same size.  The base distribution is p1.

    Returns:
      The Kullback-Liebler divergence between p1 and p2, with p1 as the base
      distribution.
    """
    return np.sum(p1 * np.log(p1 / p2))
