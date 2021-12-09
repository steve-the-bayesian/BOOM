import numpy as np
from abc import ABC, abstractmethod
import copy

"""
Wrapper classes to encapsulate and expand models and prior distributions
from the Boom library.
"""


class DoubleModel(ABC):
    """
    A base class that marks its children as being able to produce a
    boom.DoubleModel, which is simply a model that implements a 'logp' method
    measuring a real valued random variable.
    """

    @abstractmethod
    def boom(self):
        """
        Return a boom.DoubleModel with parameters set from this object.
        """

    @property
    @abstractmethod
    def mean(self):
        """
        The mean of the distribution.
        """


class SdPrior(DoubleModel):
    """A prior distribution for a standard deviation 'sigma'.  This prior assumes
    that 1/sigma**2 ~ Gamma(a, b), where a = df/2 and b = ss/2.  Here 'df' is
    the 'sample_size' and ss is the "sum of squares" equal to the sample size
    times 'sigma_guess'**2.

    This prior allows an upper limit on the support of sigma, which is infinite
    by default.

    """

    def __init__(self, sigma_guess, sample_size=.01, initial_value=None,
                 fixed=False, upper_limit=np.inf):
        """
        Create an SdPrior.

        Args:
          sigma_guess:  Guess at the value of the standard deviation.
          sample_size: Number of observations worth of information with which
            to weight the guess.
          initial_value: The initial value to be used in an MCMC chain.  This
            is not always respected.  The default value is sigma_guess.
          fixed: Flag indicating whether the parameter should be held fixed in
            an MCMC algorithm.  This is mainly for debugging and is not always
            respected.
          upper_limit: Upper limit on the value of 'sigma'.
        """
        self.sigma_guess = float(sigma_guess)
        self.sample_size = float(sample_size)
        if initial_value is None:
            initial_value = sigma_guess
        self.initial_value = float(initial_value)
        self.fixed = bool(fixed)
        self.upper_limit = float(upper_limit)

    @property
    def sum_of_squares(self):
        return self.sigma_guess**2 * self.sample_size

    def create_chisq_model(self):
        return self.boom()

    def boom(self):
        """
        Return the boom.ChisqModel corresponding to the input parameters.
        """
        import BayesBoom.boom as boom
        return boom.ChisqModel(self.sample_size, self.sigma_guess)

    @property
    def mean(self):
        """
        The mean of the distribution on the precision scale.
        """
        return self.sample_size / self.sigma_guess**2

    def __repr__(self):
        ans = f"SdPrior with sigma_guess = {self.sigma_guess}, "
        ans += f"sample_size = {self.sample_size}, "
        ans += f"upper_limit = {self.upper_limit}"
        return ans

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, payload):
        self.__dict__ = payload


class NormalPrior(DoubleModel):
    """
    A scalar normal prior distribution.
    """
    def __init__(self,
                 mu: float = 0.0,
                 sigma: float = 1.0,
                 initial_value: float = None):
        self.mu = float(mu)
        self.sigma = float(sigma)
        if initial_value is None:
            self.initial_value = mu
        else:
            self.initial_value = float(initial_value)

    @property
    def mean(self):
        return self.mu

    @property
    def sd(self):
        return self.sigma

    @property
    def variance(self):
        return self.sigma ** 2

    def boom(self):
        """
        Return the boom.GaussianModel corresponding to the object's parameters.
        """
        import BayesBoom.boom as boom
        return boom.GaussianModel(self.mu, self.sigma)

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, payload):
        self.__dict__ = payload


class Ar1CoefficientPrior(DoubleModel):
    """
    Contains the information needed to create a prior distribution on an AR1
    coefficient.
    """
    def __init__(self,
                 mu: float = 0.0,
                 sigma: float = 1.0,
                 force_stationary: bool = True,
                 force_positive: bool = False,
                 initial_value: float = None):
        """
        Args:
          mu: The prior mean of the coefficient.
          sigma:  The prior standard deviation of the coefficient.
          force_stationary: If True then the prior support for the AR1
            coefficient will be truncated to (-1, 1).
          force_positive: If True then the prior for the AR1 coefficient will
            be truncated to positive values.
          initial_value: A suggestion about where to start an MCMC sampling
            run.  The default is to use mu.
        """
        self.mu = mu
        self.sigma = sigma
        self.force_stationary = force_stationary
        self.force_positive = force_positive
        self.initial_value = initial_value
        if initial_value is None:
            self.initial_value = mu

    def boom(self):
        """
        Return the boom.GaussianModel corresponding to this object's
        parameters.
        """
        import BayesBoom.boom as boom
        return boom.GaussianModel(self.mu, self.sigma)

    @property
    def mean(self):
        return self.mu

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, payload):
        self.__dict__ = payload


class MvnPrior:
    """
    Encodes a multivariate normal distribution.
    """
    def __init__(self, mu, Sigma):
        if len(mu.shape) != 1:
            raise Exception("mu must be a vector.")
        if len(Sigma.shape) != 2:
            raise Exception("Sigma must be a matrix.")
        if Sigma.shape[0] != Sigma.shape[1]:
            raise Exception("Sigma must be symmetric")
        if Sigma.shape[0] != len(mu):
            raise Exception("mu and Sigma must be the same dimension.")
        self._mu = mu
        self._Sigma = Sigma

    @property
    def dim(self):
        return len(self._mu)

    @property
    def mu(self):
        return self._mu

    @property
    def mean(self):
        return self.mu

    @property
    def Sigma(self):
        return self._Sigma

    @property
    def variance(self):
        return self.Sigma

    def boom(self):
        """
        Return the boom.MvnModel corresponding to this object's parameters.
        """
        import BayesBoom.boom as boom
        return boom.MvnModel(boom.Vector(self._mu),
                             boom.SpdMatrix(self._Sigma))


class MvnGivenSigma:
    """
    Encodes a conditional multivariate normal distribution given an external
    variance matrix Sigma.  This model describes y ~ Mvn(mu, Sigma / kappa).
    """
    def __init__(self, mu: np.ndarray, sample_size: float):
        self._mu = np.array(mu, dtype="float").ravel()
        self._sample_size = float(sample_size)

    @property
    def dim(self):
        return len(self._mu)

    def boom(self):
        import BayesBoom.boom as boom
        return boom.MvnGivenSigma(self._mu, self._sample_size)


class UniformPrior(DoubleModel):
    """
    Univariate uniform distribution.
    """
    def __init__(self, lo, hi):
        if hi < lo:
            lo, hi = hi, lo
        self._lo = lo
        self._hi = hi

    @property
    def mean(self):
        return .5 * (self._lo + self._hi)

    def boom(self):
        """
        Return the boom.UniformModel corresponding to this object's parameters.
        """
        import BayesBoom.boom as boom
        return boom.UniformModel(self._lo, self._hi)


class DirichletPrior:
    """
    A Dirichlet prior distribution over discrete probability distributions.
    """

    def __init__(self, counts):
        counts = np.array(counts)
        if not np.all(counts > 0):
            raise Exception("All elements of 'counts' must be positive.")
        self._counts = counts

    def boom(self):
        import BayesBoom.boom as boom
        return boom.DirichletModel(boom.Vector(self._counts))


class WishartPrior:
    def __init__(self, df: float, variance_estimate: np.ndarray):
        """
        Args:
          df: The prior sample size.  For the distribution to be proper df must
            be larger than the number of rows in 'variance_estimate'.
          variance_estimate: A symmetric positive definite matrix defining the
            center of the distribution.

        Let X_i ~ Mvn(0, V).  Then the Wishart(nu, V) distribution describes
        the sum of 'nu' draws X_i * X_i'.  If the draws are placed as rows in a
        matrix X then X'X ~ Wishart(nu, V).  The mean of this distribution is
        nu * V.

        The Wishart distribution is the conjugate prior for the precision
        parameter (inverse variance) of the multivariate normal distribution.
        """
        sumsq = df * variance_estimate
        if len(sumsq.shape) != 2:
            raise Exception("sumsq must be a matrix")

        if sumsq.shape[0] != sumsq.shape[1]:
            raise Exception("sumsq must be square")

        sym_sumsq = (sumsq + sumsq.T) * .5
        sumabs = np.sum(np.abs(sumsq - sym_sumsq))
        relative = np.sum(np.abs(sumsq))
        if sumabs / relative > 1e-8:
            raise Exception("sumsq must be symmetric")

        if df <= sumsq.shape[0]:
            raise Exception(
                "df must be largern than nrow(sumsq) for the prior to be "
                "proper.")

        self._df = df
        self._sumsq = sumsq

    @property
    def variance_estimate(self):
        return self._sumsq / self._df

    @property
    def df(self):
        return self._df

    def boom(self):
        import BayesBoom.boom as boom
        return boom.WishartModel(self.df, self.variance_estimate)


class GaussianSuf:
    """
    Sufficient statistics for a scalar normal model.
    """

    def __init__(self, data=None):
        """
        Args:
          data: If None (the default) then an empty GaussianSuf is created.
            Otherwise create a new GaussianSuf summarizing 'data'.
        """
        self._sum = 0
        self._sumsq = 0
        self._n = 0
        if data is not None:
            self.update(data)

    def update(self, incremental_data):
        """
        Add summaries of the incremental data to the data already summarized.

        Args:
          data:  A 1-d numpy array, or equivalent.

        Effects:
          The sufficient statistics in the object are updated to describe data.
        """
        y = np.array(incremental_data)
        self._sum += np.nansum(y)
        self._sumsq += np.nansum(y * y)
        self._n += np.sum(~np.isnan(incremental_data))

    def combine(self, other):
        """
        Add the sufficient statistics from 'other' to 'self'.  This operation is
        done inplace.  The 'other' object is unaffected.
        """
        self._n += other._n
        self._sum += other._sum
        self._sumsq += other._sumsq

    def __iadd__(self, other):
        """
        Implements operator +=.  Other can either be a GaussianSuf or raw data.
        """
        if isinstance(other, GaussianSuf):
            self.combine(other)
        else:
            self.update(other)
        return self

    def __add__(self, other):
        """
        Implements operator+.  Other can either be a GaussianSuf or raw data.
        """
        ans = copy.copy(self)
        ans += other
        return ans

    @property
    def sample_size(self):
        return self._n

    @property
    def sum(self):
        return self._sum

    @property
    def mean(self):
        if self.sample_size > 0:
            return self._sum / self.sample_size
        else:
            return 0.0

    @property
    def sumsq(self):
        return self._sumsq

    def centered_sumsq(self, center=None):
        if self.sample_size <= 0:
            return 0
        if center is None:
            center = self.mean
        n = self.sample_size
        return self.sumsq - 2 * center * self.sum + n * center ** 2

    @property
    def sample_sd(self):
        return np.sqrt(self.sample_variance)

    @property
    def sample_variance(self):
        n = self.sample_size
        if n < 2:
            return 0
        return self.centered_sumsq() / (n - 1)
