"""
Python wrapper for boom.BinomialLogitModel (logistic regression).
"""

import numpy as np

from ..boom_utils import to_boom_vector, to_boom_matrix, to_boom_spd


class BinomialLogitMvnPrior:
    """
    Multivariate normal prior on logistic regression coefficients,
    leading to a boom.BinomialLogitAuxmixSampler posterior sampler.

      beta  ~  N(mu, Sigma)

    Args:
      mu: Prior mean vector.  Defaults to the zero vector (inferred from
        xdim at sampler-creation time).
      Sigma: Prior covariance matrix.  Defaults to variance_scale * I.
      variance_scale: Scalar applied to the identity when Sigma is None.
        Ignored if Sigma is provided explicitly.
      clt_threshold: Minimum number of trials per observation before the
        CLT-based approximation is used in BinomialLogitAuxmixSampler.
    """

    def __init__(self, mu=None, Sigma=None, variance_scale: float = 1.0,
                 clt_threshold: int = 10):
        self._mu = np.asarray(mu, dtype=float) if mu is not None else None
        self._Sigma = np.asarray(Sigma, dtype=float) if Sigma is not None else None
        self._variance_scale = float(variance_scale)
        self._clt_threshold = int(clt_threshold)

    def create_sampler(self, boom_model):
        """
        Build and return a boom.BinomialLogitAuxmixSampler.

        Args:
          boom_model: The boom.BinomialLogitModel to be sampled.
        """
        import BayesBoom.boom as boom

        xdim = boom_model.xdim
        mu = self._mu if self._mu is not None else np.zeros(xdim)
        Sigma = (self._Sigma if self._Sigma is not None
                 else np.eye(xdim) * self._variance_scale)

        mvn = boom.MvnModel(to_boom_vector(mu), to_boom_spd(Sigma))

        return boom.BinomialLogitAuxmixSampler(
            boom_model, mvn, self._clt_threshold, boom.GlobalRng.rng)


class BinomialLogitSpikeSlabPrior:
    """
    Spike-and-slab prior for logistic regression, producing a
    boom.BinomialLogitSpikeSlabSampler posterior sampler.

    Each coefficient has an independent Bernoulli inclusion indicator.
    When included, the coefficient follows the specified multivariate
    normal slab.

    Args:
      mu: Slab prior mean.  Defaults to zeros.
      Sigma: Slab prior covariance.  Defaults to variance_scale * I.
      variance_scale: Scalar applied to the identity when Sigma is None.
      expected_model_size: Prior expected number of non-zero coefficients.
        Sets inclusion probability p = min(1, expected_model_size / xdim).
      clt_threshold: Minimum trials per observation for the CLT approximation
        inside the sampler.
    """

    def __init__(self, mu=None, Sigma=None, variance_scale: float = 1.0,
                 expected_model_size: float = 1.0, clt_threshold: int = 5):
        self._mu = np.asarray(mu, dtype=float) if mu is not None else None
        self._Sigma = np.asarray(Sigma, dtype=float) if Sigma is not None else None
        self._variance_scale = float(variance_scale)
        self._expected_model_size = float(expected_model_size)
        self._clt_threshold = int(clt_threshold)

    def create_sampler(self, boom_model):
        """
        Build and return a boom.BinomialLogitSpikeSlabSampler.

        Args:
          boom_model: The boom.BinomialLogitModel to be sampled.
        """
        import BayesBoom.boom as boom

        xdim = boom_model.xdim
        mu = self._mu if self._mu is not None else np.zeros(xdim)
        Sigma = (self._Sigma if self._Sigma is not None
                 else np.eye(xdim) * self._variance_scale)

        slab = boom.MvnModel(to_boom_vector(mu), to_boom_spd(Sigma))

        prob = min(1.0, self._expected_model_size / xdim)
        spike = boom.VariableSelectionPrior(
            to_boom_vector(np.full(xdim, prob)))

        return boom.BinomialLogitSpikeSlabSampler(
            boom_model, slab, spike, self._clt_threshold, boom.GlobalRng.rng)


class BinomialLogitModel:
    """
    Python wrapper for boom.BinomialLogitModel.

    Each observation (y_i, n_i, x_i) follows:

      y_i | n_i, x_i, beta  ~  Binomial(n_i, p_i)
      logit(p_i) = x_i @ beta

    For binary (Bernoulli) data, set trials to 1 (the default).

    Maintains the data, a prior on beta, and a posterior sampler as
    private attributes.  Call boom() to get a fully configured C++ model
    ready for MCMC.

    Typical use::

        model = BinomialLogitModel(X, y)            # binary 0/1 response
        draws = np.zeros((niter, model.xdim))
        for i in range(niter):
            model.sample_posterior()
            draws[i] = model.coefficients

    For binomial (grouped) data::

        model = BinomialLogitModel(X, y=successes, trials=trials)

    For spike-and-slab variable selection::

        prior = BinomialLogitSpikeSlabPrior(expected_model_size=3)
        model = BinomialLogitModel(X, y, prior=prior)
    """

    def __init__(self, X, y, trials=None, prior=None):
        """
        Args:
          X: Predictor matrix (numpy array or DataFrame), shape (n, p).
             Include an explicit column of 1's for the intercept.
          y: Success counts.  0/1 for binary data; non-negative integers
             for grouped binomial data.
          trials: Trial counts per observation.  Defaults to all-ones
             (Bernoulli / binary data).
          prior: A prior object with a ``create_sampler(boom_model)``
             method — either BinomialLogitMvnPrior (default) or
             BinomialLogitSpikeSlabPrior.  For backward compatibility, a
             bare MvnPrior from BayesBoom.models is also accepted and
             treated as a BinomialLogitMvnPrior with the same distribution.
        """
        self._X = np.asarray(X, dtype=float)
        if self._X.ndim == 1:
            self._X = self._X.reshape(-1, 1)

        self._y = np.asarray(y, dtype=float).ravel()

        if trials is None:
            self._trials = np.ones(len(self._y))
        else:
            self._trials = np.asarray(trials, dtype=float).ravel()

        if len(self._y) != self._X.shape[0]:
            raise ValueError(
                f"len(y) = {len(self._y)} != nrow(X) = {self._X.shape[0]}.")
        if len(self._trials) != len(self._y):
            raise ValueError("len(trials) must equal len(y).")

        self._prior = prior
        self._boom_model = None
        self._boom_sampler = None

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def boom(self):
        """
        Return the boom.BinomialLogitModel.

        On the first call, creates the C++ model, populates it with the
        stored data, builds and attaches the posterior sampler, then
        returns the model.  Subsequent calls return the cached model.
        """
        if self._boom_model is not None:
            return self._boom_model

        import BayesBoom.boom as boom

        xdim = self._X.shape[1]
        self._boom_model = boom.BinomialLogitModel(xdim, True)

        self._boom_model.add_dataset(
            to_boom_vector(self._y),
            to_boom_vector(self._trials),
            to_boom_matrix(self._X))

        if self._prior is None:
            self._prior = BinomialLogitMvnPrior()

        if hasattr(self._prior, 'create_sampler'):
            self._boom_sampler = self._prior.create_sampler(self._boom_model)
        else:
            # Legacy: bare MvnPrior passed directly
            self._boom_sampler = boom.BinomialLogitAuxmixSampler(
                self._boom_model, self._prior.boom(), 10, boom.GlobalRng.rng)

        self._boom_model.set_method(self._boom_sampler)

        return self._boom_model

    def sample_posterior(self):
        """Execute one MCMC draw of beta."""
        self.boom().sample_posterior()

    def add_data(self, y, X, trials=None):
        """
        Add observations to the model.

        Can be called before or after boom().  When called after boom(),
        the data is added directly to the live C++ model object.

        Args:
          y: Success counts for the new observations.
          X: Predictor matrix for the new observations.
          trials: Trial counts.  Defaults to all-ones (binary data).
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if trials is None:
            trials = np.ones(len(y))
        else:
            trials = np.asarray(trials, dtype=float).ravel()

        self._X = np.vstack([self._X, X])
        self._y = np.concatenate([self._y, y])
        self._trials = np.concatenate([self._trials, trials])

        if self._boom_model is not None:
            self._boom_model.add_dataset(
                to_boom_vector(y),
                to_boom_vector(trials),
                to_boom_matrix(X))

    # ------------------------------------------------------------------
    # Parameter access
    # ------------------------------------------------------------------

    @property
    def xdim(self) -> int:
        """Number of columns in the predictor matrix."""
        return self._X.shape[1]

    @property
    def coefficients(self) -> np.ndarray:
        """Current logistic regression coefficients as a numpy array."""
        return self.boom().Beta.to_numpy()

    # ------------------------------------------------------------------
    # Data / prior access
    # ------------------------------------------------------------------

    @property
    def prior(self):
        """The prior distribution on beta."""
        return self._prior

    @property
    def sample_size(self) -> int:
        """Number of observations in the dataset."""
        return len(self._y)
