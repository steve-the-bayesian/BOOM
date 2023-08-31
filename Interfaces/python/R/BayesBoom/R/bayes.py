import numpy as np
from abc import ABC, abstractmethod
import copy

from .boom_py_utils import to_boom_vector, to_boom_spd

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

    def __eq__(self, other):
        return self.__dict__ == other.__dict__


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


class GammaModel(DoubleModel):
    def __init__(self, shape=None, scale=None, mu=None, a=None, b=None):
        """
        A GammaModel(a, b) can be defined either in terms of its shape (a) and
        scale (b) paramaeters (with mean a/b, variance a/b^2), or it's mean
        (mu) and shape parameters (so the mean is mu and the variance is
        mu^2/a).

        Args:
          shape:  The shape parameter a.
          scale:  The scale parameter b.
          mu:  The mean of the distribution.
          a:  Another name for the shape parameter.
          b:  Another name for the scale parameter.

        Only two of these parameters need to be specified.  If all three are
        given, then 'mu' is ignored.
        """
        if a is not None:
            shape = a
        if b is not None:
            scale = b

        if (shape is None) + (scale is None) + (mu is None) > 1:
            raise Exception("Two parameters must be specified.")

        self._a = shape
        if self._a is None:
            self._a = scale * mu

        self._b = scale
        if self._b is None:
            self._b = mu / shape

        if self._a <= 0 or self._b <= 0:
            raise Exception("GammaModel parameters must be positive.")

    @property
    def mean(self):
        return self._a / self._b

    @property
    def variance(self):
        return self._a / self._b**2

    @property
    def a(self):
        return self._a

    @property
    def shape(self):
        return self._a

    @property
    def b(self):
        return self._b

    @property
    def scale(self):
        return self._b

    def boom(self):
        import BayesBoom.boom as boom
        return boom.GammaModel(self.a, self.b)

    def __repr__(self):
        ans = f"A GammaModel with shape = {self.shape} "
        ans += f"and scale = {self.scale}."
        return ans


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


class MvnBase(ABC):
    @property
    @abstractmethod
    def dim(self):
        """
        The dimension of the variable described by the distribution.
        """
    @property
    @abstractmethod
    def mean(self):
        """
        The mean of the distribution.
        """

    @property
    @abstractmethod
    def variance(self):
        """
        The variance of the distribution, as a 2-d numpy array.
        """

    @abstractmethod
    def boom(self):
        """
        Return the corresponding boom object.
        """


class MvnPrior(MvnBase):
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


class MvnGivenSigma(MvnBase):
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
        return boom.MvnGivenSigma(to_boom_vector(self._mu),
                                  self._sample_size)

    def __eq__(self, other):
        return self.__dict__ == other.__dict__


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


class BetaPrior(DoubleModel):
    """
    A distribution, typically used as the prior over a scalar probability.
    """
    def __init__(self, a=1.0, b=1.0):
        self._a = float(a)
        self._b = float(b)

    @property
    def mean(self):
        return self._a / (self._a + self._b)

    def boom(self):
        import BayesBoom.boom as boom
        return boom.BetaModel(self._a, self._b)

    def __eq__(self, other):
        return self.__dict__ == other.__dict__


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

    def __eq__(self, other):
        return self.__dict__ == other.__dict__


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

    def __eq__(self, other):
        return self.__dict__ == other.__dict__


class RegSuf:
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
        return boom.RegSuf(xtx=R.to_boom_spd(self._xtx),
                           xty=R.to_boom_vector(self._xty),
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


class ScottZellnerMvnPrior(MvnBase):
    """
    A Zellner prior on a set of regression coefficients, shrunk towards the
    diagonal by a parameterized amount.

    The Zellner prior is a multivariate normal distribution with mean mu and
    precision matrix A = g * X'X / sigsq, where g is a number specified by the
    modeler, and sigsq is the residual variance parameter in the regression
    model.

    The ScottZellnerPrior is a modified version of the Zellner prior with X'X
    replaced by (1 - a) * X'X + a * diag(X'X).  That is, a weighted average of
    X'X with its diagonal.

    The coefficient 'g' in the ordinary Zellner prior is replaced by
    'prior_nobs' / n where n is the sample size.  Because X'X is the total
    information from the data available in a regression problem, X'X/n is the
    average information available from a single data point.  Thus 'prior_nobs'
    can be interpreted as the number of data points worth of information you
    want the prior to weigh.

    """

    def __init__(self,
                 suf: RegSuf,
                 diagonal_shrinkage: float = .05,
                 prior_nobs: float = 1.0,
                 sigma: float = 1.0):
        """
        Args:
          suf: The sufficient statistics of the regression model.
          diagonal_shrinkage: The 'a' parameter in the class description.  The
            amount by which to shrink towards the diagonal of X'X.  A real
            number between 0 and 1.
          prior_nobs: The number of observations worth of weight to assign the
            prior.  A positive scalar.
          sigma: A scale factor for the prior.  In some applications sigma must
            be determined outside this class.  Child classes may obtain sigma
            from a callback, for example.
        """
        omega = suf.xtx
        weight = diagonal_shrinkage
        omega = (1 - weight) * omega + weight * np.diag(np.diag(omega))

        omega = omega * (prior_nobs / suf.sample_size)
        self._precision = omega / sigma**2

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
