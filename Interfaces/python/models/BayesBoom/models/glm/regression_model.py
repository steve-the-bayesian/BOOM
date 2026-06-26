"""
Python wrapper for boom.RegressionModel (Gaussian linear regression).
"""

import numpy as np

from ..boom_utils import to_boom_vector, to_boom_spd
from ..MvnModel import MvnBase


class RegressionSuf:
    """
    The sufficient statistics needed to specify a regression model.
    """

    def __init__(self, xtx, xty, sample_sd, sample_size=None, ybar=None,
                 xbar=None):
        """
        In what follows X is the design matrix of predictors, and y is the
        column vector of responses.  The matrix transpose of X is denoted X'.

        Args:
          xtx: The cross product matrix X'X.
          xty: X'y
          sample_sd:  The sample standard deviation of the y's.
          sample_size: The number of observations covered by the sufficient
            statistics.  If X contains a column of 1's in column 0 then
            sample_size can be None.

          ybar: The mean of the y's (a scalar).  If X contains a column of 1's
            in column 0 then this can be None.

          xbar: The mean of the X's (a vector).
        """
        xtx = np.array(xtx)
        xty = np.array(xty)
        xbar = np.array(xbar)

        if xtx.shape[0] != xtx.shape[1]:
            raise Exception("xtx must be square")
        if xtx.shape[0] != xty.shape[0]:
            raise Exception("xtx and xty must be the same size.")

        if not sample_sd >= 0:
            raise Exception("The sample_sd must be non-negative.")

        if sample_size is None:
            sample_size = xtx[0, 0]
        if not sample_size >= 0:
            raise Exception("The sample size must be non-negative.")

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
        X = np.array(X)
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        y = np.array(y).ravel()

        if X.shape[0] != y.shape[0]:
            raise Exception(
                f"The length of y ({len(y)}) must match the number of rows "
                f"in X ({X.shape[0]}).")

        xtx = X.T @ X
        xty = X.T @ y
        sample_size = len(y)
        sample_sd = np.std(y, ddof=1)
        ybar = np.mean(y)
        xbar = X.mean(axis=0)
        return cls(xtx, xty, sample_sd, sample_size, ybar, xbar)

    def boom(self):
        import BayesBoom.boom as boom
        import BayesBoom.R as R
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
        return self._sample_sd**2

    @property
    def sample_size(self):
        return self._sample_size


class RegressionConjugatePrior:
    """
    Normal-inverse-gamma conjugate prior for Gaussian regression.

      beta | sigma^2 ~ N(b, sigma^2 * Omega^{-1})
      1 / sigma^2   ~ Gamma(df/2, ss/2)

    The prior mean b has its intercept component set to ybar and all
    slope components set to zero.  The unscaled precision Omega is a
    scaled and ridge-shrunk version of X'X:

      Omega = kappa * [(1 - a) * X'X + a * diag(X'X)] / n

    where kappa = prior_sample_size, a = diagonal_shrinkage, n = sample size.

    Args:
      prior_sample_size: Observations-worth of weight for the coefficient
        prior (kappa above).
      diagonal_shrinkage: Weight on diag(X'X) versus full X'X in the
        precision matrix.  Keeps the prior proper even when X'X is
        rank-deficient.
      sigma_guess: Point estimate of the residual standard deviation.
        If None, estimated from the data (sample_sd of y).
      sigma_df: Degrees of freedom for the residual variance prior.
        Small values give a diffuse prior on sigma.
    """

    def __init__(self, prior_sample_size: float = 1.0,
                 diagonal_shrinkage: float = 0.05,
                 sigma_guess: float = None,
                 sigma_df: float = 0.01):
        self._prior_sample_size = float(prior_sample_size)
        self._diagonal_shrinkage = float(diagonal_shrinkage)
        self._sigma_guess = sigma_guess
        self._sigma_df = float(sigma_df)

    def create_sampler(self, boom_model):
        """
        Build and return a boom.RegressionConjugateSampler.

        Args:
          boom_model: The boom.RegressionModel to be sampled.
          suf: RegressionSuf holding the data sufficient statistics.
        """
        import BayesBoom.boom as boom

        suf = boom_model.suf
        xtx = to_numpy(suf.xtx)
        
        sigma_guess = self._sigma_guess
        if sigma_guess is None:
            sigma_guess = max(suf.sample_sd, 1e-6)

        # Unscaled prior precision: kappa * [(1-a)*xtx + a*diag(xtx)] / n
        a = self._diagonal_shrinkage
        ominv = (1.0 - a) * xtx + a * np.diag(np.diag(xtx))
        ominv *= self._prior_sample_size / suf.sample_size

        # Intercept shrinks to ybar; slopes shrink to 0
        mean = np.zeros(suf.xdim)
        mean[0] = suf.ybar

        # MvnGivenScalarSigma: beta | sigsq ~ N(mean, sigsq * ominv^{-1})
        slab = boom.MvnGivenScalarSigma(
            to_boom_vector(mean),
            to_boom_spd(ominv),
            boom_model.Sigsq_prm)

        prec_prior = boom.ChisqModel(self._sigma_df, sigma_guess)

        return boom.RegressionConjugateSampler(
            boom_model, slab, prec_prior, boom.GlobalRng.rng)


class RegressionModel:
    """
    Python wrapper for boom.RegressionModel.

      y | X, beta, sigma^2  ~  N(X @ beta, sigma^2)

    Maintains a RegressionSuf (sufficient statistics), a RegressionConjugatePrior,
    and a posterior sampler as private attributes.  Call boom() to get a
    fully-configured C++ model ready for MCMC.

    Typical use::

        model = RegressionModel(X, y)
        draws = np.zeros((niter, model.xdim))
        sigma_draws = np.zeros(niter)
        for i in range(niter):
            model.sample_posterior()
            draws[i] = model.coefficients
            sigma_draws[i] = model.sigma
    """

    def __init__(self, X=None, y=None, suf: RegressionSuf = None, prior=None):
        """
        Args:
          X: Predictor matrix (numpy array or pandas DataFrame).  Include an
             explicit column of 1's for the intercept.
          y: Response vector.
          suf: A RegressionSuf object.  Supply this instead of (X, y) when only
             sufficient statistics are available.
          prior: A prior object with a ``create_sampler(boom_model)``
             method — either RegressionConjugatePrior (default) or
             RegressionSpikeSlabPrior.  If None, RegressionConjugatePrior
             with default settings is used.
        """
        if suf is not None:
            self._suf = suf
        elif X is not None and y is not None:
            self._suf = RegressionSuf.from_data(X, y)
        else:
            raise ValueError("Provide either (X, y) or suf.")

        self._prior = prior
        self._boom_model = None
        self._boom_sampler = None

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def boom(self):
        """
        Return the boom.RegressionModel.

        On the first call, creates the C++ model from the stored sufficient
        statistics, builds and attaches the posterior sampler, then returns
        the model.  Subsequent calls return the cached model.
        """
        if self._boom_model is not None:
            return self._boom_model

        import BayesBoom.boom as boom

        self._boom_model = boom.RegressionModel(self._suf.boom())

        if self._prior is None:
            self._prior = RegressionConjugatePrior()

        self._boom_sampler = self._prior.create_sampler(
            self._boom_model, self._suf)
        self._boom_model.set_method(self._boom_sampler)

        return self._boom_model

    def sample_posterior(self):
        """Execute one MCMC draw of (beta, sigma)."""
        self.boom().sample_posterior()

    # ------------------------------------------------------------------
    # Parameter access
    # ------------------------------------------------------------------

    @property
    def xdim(self) -> int:
        """Number of columns in the predictor matrix."""
        return self._suf.xdim

    @property
    def coefficients(self) -> np.ndarray:
        """Current regression coefficients as a numpy array."""
        return self.boom().Beta.to_numpy()

    @property
    def sigma(self) -> float:
        """Current residual standard deviation."""
        return self.boom().sigma

    @property
    def residual_variance(self) -> float:
        return self.sigma ** 2

    def log_likelihood(self) -> float:
        return self.boom().log_likelihood()

    # ------------------------------------------------------------------
    # Data / prior access
    # ------------------------------------------------------------------

    @property
    def suf(self) -> RegressionSuf:
        """The sufficient statistics for this model."""
        return self._suf

    @property
    def prior(self):
        """The prior distribution on (beta, sigma)."""
        return self._prior


