"""
Python wrappers for BayesBoom C++ objects in BOOM/Models/Glm.

Covered so far:
  RegressionModel  →  RegSuf  (sufficient statistics)
  RegressionSlabPrior-style  →  ScottZellnerMvnPrior
"""

import numpy as np

from ..boom_utils import to_boom_vector, to_boom_spd
from ..MvnModel import MvnBase


class RegSuf:
    """Sufficient statistics for a Gaussian linear regression model."""

    def __init__(self, xtx, xty, sample_sd, sample_size=None, ybar=None,
                 xbar=None):
        """
        Args:
          xtx: Cross product matrix X'X.
          xty: X'y.
          sample_sd: Sample standard deviation of y.
          sample_size: Number of observations (inferred from xtx[0,0] if None
            and X has an intercept column).
          ybar: Mean of y (inferred from xty[0]/n if X has an intercept).
          xbar: Mean of X columns.
        """
        xtx = np.array(xtx)
        xty = np.array(xty)
        xbar = np.array(xbar)

        if xtx.shape[0] != xtx.shape[1]:
            raise Exception("xtx must be square.")
        if xtx.shape[0] != xty.shape[0]:
            raise Exception("xtx and xty must have matching dimensions.")
        if not sample_sd >= 0:
            raise Exception("sample_sd must be non-negative.")

        if sample_size is None:
            sample_size = xtx[0, 0]
        if not sample_size >= 0:
            raise Exception("sample_size must be non-negative.")

        if xbar is None:
            raise Exception("xbar must be supplied.")
        if xbar.shape[0] != xty.shape[0]:
            raise Exception("xbar has the wrong size.")

        if ybar is None:
            ybar = xty[0] / sample_size

        self._xtx = xtx
        self._xty = xty
        self._sample_sd = sample_sd
        self._sample_size = sample_size
        self._ybar = ybar
        self._xbar = xbar

    @classmethod
    def from_data(cls, X, y):
        """Construct from raw data arrays."""
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        y = np.array(y).ravel()
        if X.shape[0] != y.shape[0]:
            raise Exception(
                f"len(y) = {len(y)} does not match nrow(X) = {X.shape[0]}.")
        xtx = X.T @ X
        xty = X.T @ y
        sample_size = len(y)
        sample_sd = np.std(y, ddof=1)
        ybar = np.mean(y)
        xbar = X.mean(axis=0)
        return cls(xtx, xty, sample_sd, sample_size, ybar, xbar)

    def boom(self):
        import BayesBoom.boom as boom
        return boom.RegSuf(xtx=to_boom_spd(self._xtx),
                           xty=to_boom_vector(self._xty),
                           sample_sd=self._sample_sd,
                           sample_size=self._sample_size,
                           ybar=self._ybar,
                           xbar=self._xbar)

    @property
    def xtx(self):
        return self._xtx

    @property
    def xty(self):
        return self._xty

    @property
    def xdim(self):
        return self._xtx.shape[0]

    @property
    def xbar(self):
        return self._xbar

    @property
    def mean_x(self):
        return self._xbar

    @property
    def mean_y(self):
        return self._ybar

    @property
    def ybar(self):
        return self._ybar

    @property
    def sample_sd(self):
        return self._sample_sd

    @property
    def sample_variance(self):
        return self._sample_sd ** 2

    @property
    def sample_size(self):
        return self._sample_size


class ScottZellnerMvnPrior(MvnBase):
    """
    Zellner-style MVN prior for regression coefficients, shrunk toward the
    diagonal of X'X.

    Precision = g * [(1-a) * X'X + a * diag(X'X)] / sigma^2

    where g = prior_nobs / n.  The prior mean is zero except for the intercept
    term, which is set to ybar.

    Args:
      suf: Sufficient statistics of the regression model.
      diagonal_shrinkage: Weight 'a' on diag(X'X) vs full X'X.  In [0, 1].
      prior_nobs: Number of observations-worth of weight for the prior.
      sigma: Scale factor (typically the residual standard deviation).
    """

    def __init__(self,
                 suf: RegSuf,
                 diagonal_shrinkage: float = .05,
                 prior_nobs: float = 1.0,
                 sigma: float = 1.0):
        omega = suf.xtx
        weight = diagonal_shrinkage
        omega = (1 - weight) * omega + weight * np.diag(np.diag(omega))
        omega = omega * (prior_nobs / suf.sample_size)
        self._precision = omega / sigma ** 2
        self._mean = np.zeros(omega.shape[0])
        self._mean[0] = suf.ybar
        self._variance = None

    @property
    def dim(self):
        return self._mean.shape[0]

    @property
    def mean(self):
        return self._mean

    @property
    def variance(self):
        if self._variance is None:
            self._variance = np.linalg.inv(self._precision)
        return self._variance

    def boom(self):
        import BayesBoom.boom as boom
        return boom.MvnModel(to_boom_vector(self.mean),
                             to_boom_spd(self.variance))
