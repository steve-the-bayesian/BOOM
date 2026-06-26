"""
Python wrapper for boom.RegressionModel (Gaussian linear regression).
"""

import numpy as np

from ..boom_utils import to_boom_vector, to_boom_spd
from .glm import RegSuf


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

    def create_sampler(self, boom_model, suf: RegSuf):
        """
        Build and return a boom.RegressionConjugateSampler.

        Args:
          boom_model: The boom.RegressionModel to be sampled.
          suf: RegSuf holding the data sufficient statistics.
        """
        import BayesBoom.boom as boom

        sigma_guess = self._sigma_guess
        if sigma_guess is None:
            sigma_guess = max(suf.sample_sd, 1e-6)

        # Unscaled prior precision: kappa * [(1-a)*xtx + a*diag(xtx)] / n
        xtx = np.array(suf.xtx, dtype=float)
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


class RegressionSpikeSlabPrior:
    """
    Spike-and-slab prior for Gaussian regression, producing a
    boom.BregVsSampler posterior sampler.

    Each coefficient has an independent Bernoulli inclusion indicator.
    When included, the coefficient follows the same MvnGivenScalarSigma
    slab used by RegressionConjugatePrior.  The residual variance has a
    scaled-chi-squared prior.

    Args:
      expected_model_size: Prior expected number of non-zero coefficients.
        Used to set inclusion probability p = min(1, expected_model_size / xdim).
      prior_sample_size: Observations-worth of weight on the slab (kappa).
      diagonal_shrinkage: Ridge toward diag(X'X) in the slab precision.
      sigma_guess: Prior guess for residual standard deviation.
        Estimated from data if None.
      sigma_df: Degrees of freedom for the residual variance prior.
    """

    def __init__(self, expected_model_size: float = 1.0,
                 prior_sample_size: float = 1.0,
                 diagonal_shrinkage: float = 0.05,
                 sigma_guess: float = None,
                 sigma_df: float = 0.01):
        self._expected_model_size = float(expected_model_size)
        self._prior_sample_size = float(prior_sample_size)
        self._diagonal_shrinkage = float(diagonal_shrinkage)
        self._sigma_guess = sigma_guess
        self._sigma_df = float(sigma_df)

    def create_sampler(self, boom_model, suf: RegSuf):
        """
        Build and return a boom.BregVsSampler.

        Args:
          boom_model: The boom.RegressionModel to be sampled.
          suf: RegSuf holding the data sufficient statistics.
        """
        import BayesBoom.boom as boom

        sigma_guess = self._sigma_guess
        if sigma_guess is None:
            sigma_guess = max(suf.sample_sd, 1e-6)

        xtx = np.array(suf.xtx, dtype=float)
        a = self._diagonal_shrinkage
        ominv = (1.0 - a) * xtx + a * np.diag(np.diag(xtx))
        ominv *= self._prior_sample_size / suf.sample_size

        mean = np.zeros(suf.xdim)
        mean[0] = suf.ybar

        # BregVsSampler requires MvnGivenScalarSigma (not generic MvnBase)
        slab = boom.MvnGivenScalarSigma(
            to_boom_vector(mean),
            to_boom_spd(ominv),
            boom_model.Sigsq_prm)

        xdim = suf.xdim
        prob = min(1.0, self._expected_model_size / xdim)
        spike = boom.VariableSelectionPrior(
            to_boom_vector(np.full(xdim, prob)))

        prec_prior = boom.ChisqModel(self._sigma_df, sigma_guess)

        return boom.BregVsSampler(
            boom_model, slab, prec_prior, spike, boom.GlobalRng.rng)


class RegressionModel:
    """
    Python wrapper for boom.RegressionModel.

      y | X, beta, sigma^2  ~  N(X @ beta, sigma^2)

    Maintains a RegSuf (sufficient statistics), a RegressionConjugatePrior,
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

    def __init__(self, X=None, y=None, suf: RegSuf = None, prior=None):
        """
        Args:
          X: Predictor matrix (numpy array or pandas DataFrame).  Include an
             explicit column of 1's for the intercept.
          y: Response vector.
          suf: A RegSuf object.  Supply this instead of (X, y) when only
             sufficient statistics are available.
          prior: A prior object with a ``create_sampler(boom_model, suf)``
             method — either RegressionConjugatePrior (default) or
             RegressionSpikeSlabPrior.  If None, RegressionConjugatePrior
             with default settings is used.
        """
        if suf is not None:
            self._suf = suf
        elif X is not None and y is not None:
            self._suf = RegSuf.from_data(X, y)
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
    def suf(self) -> RegSuf:
        """The sufficient statistics for this model."""
        return self._suf

    @property
    def prior(self):
        """The prior distribution on (beta, sigma)."""
        return self._prior
