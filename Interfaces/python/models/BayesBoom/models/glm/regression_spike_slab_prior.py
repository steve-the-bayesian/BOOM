import numpy as np
import BayesBoom.boom as boom
from .regression_model import RegressionSuf
from ..boom_utils import to_boom_vector, to_boom_matrix, to_boom_spd
from ..MvnModel import MvnBase

class RegressionSlabPrior:
    """
    A multivariate normal distribution intended to be the prior in a multiple
    regression problem.  The prior is

    beta ~ N(b, V)

    where b = (ybar, 0, 0, 0, ....)
    and V^{-1} = kappa * [(1 - alpha) * xtx + alpha * diag(xtx)] / n

    The mean parameter shrinks the intercept towards the sample mean, and all
    other coefficients towards zero.  In the literature it is more standard to
    shrink all coefficients towards zero, but in practice this can inflate
    estimates of the residual standard deviation.

    The prior precision is defined in terms of xtx: the cross product matrix
    from the regression problem.  We average xtx with its diagonal (with weight
    alpha on the diagonal) to ensure that the overall matrix is full rank.  xtx
    is the information matrix for the regression coefficients in a standard
    regression problem, so dividing by 'n' (the sample size) turns the whole
    thing into the "average information from a single observation."
    Multiplying by 'kappa' means that the information content of the prior is
    equivalent to 'kappa' prior observations.
    """

    def __init__(self,
                 xtx,
                 sample_mean,
                 data_sample_size,
                 prior_sample_size=1.0,
                 diagonal_shrinkage=0.05):
        """
        Args:
          Please see the class comments, above.
          xtx:  The cross product matrix from the regression.
          sample_mean: The mean of the response variable in the regression
            problem ('ybar' above).
          data_sample_size: The number of observations in the regression ('n'
            above).
          prior_sample_size: The number of observations of prior weight to
            assign the prior.  ('kappa' above).
          diagonal_shrinkage: The weight to assign the diagonal of xtx in the
            full rank adjustment.  ('alpha' above).
        """
        self._xtx = xtx
        self._sample_mean = sample_mean
        self._data_sample_size = data_sample_size
        self._prior_sample_size = prior_sample_size
        self._diagonal_shrinkage = diagonal_shrinkage

    def set_xtx(self, xtx: np.ndarray):
        self._xtx = xtx

    def boom(self, sigsq_param: boom.UnivParams):
        xtx = self._xtx if self._xtx is not None else np.ones((1, 1))
        return boom.RegressionSlabPrior(
            boom.SpdMatrix(xtx),
            sigsq_param,
            self._sample_mean,
            self._data_sample_size,
            self._prior_sample_size,
            self._diagonal_shrinkage)


class RegressionSpikeSlabPrior:
    """
    Components of spike and slab priors that are shared regardless of the model
    type.
    """
    def __init__(self,
                 x,
                 y=None,
                 expected_r2=.5,
                 prior_df=.01,
                 expected_model_size=1,
                 prior_information_weight=.01,
                 diagonal_shrinkage=.5,
                 optional_coefficient_estimate=None,
                 max_flips=-1,
                 mean_y=None,
                 sdy=None,
                 prior_inclusion_probabilities=None,
                 sigma_upper_limit=None,
                 max_size=None):
        """
        Computes information that is shared by the different implementation of
        spike and slab priors.  Currently, the only difference between the
        different priors is the prior variance on the regression coefficients.
        When that changes, change this class accordingly, and change all the
        classes that inherit from it.

        Args:
          x: Either the design matrix (as a pd.DataFrame or a np.array), or an
            object of class glm.RegressionSuf containing the sufficient
            statistics for the model to be fit.
          y: The response vector (as a pd.Series or a np.array).  If 'x'
             contains model sufficient statistics then y is not used.
          expected_r2: The R^2 statistic that the model is expected
            to achieve.  Used along with 'sdy' to derive a prior distribution
            for the residual variance.
          prior_df: The number of observations worth of weight to give to the
            guess at the residual variance.
          expected_model_size: The expected number of nonzero coefficients in
            the model.  Used to set prior_inclusion_probabilities to
            expected_model_size / number_of_variables.  If expected_model_size
            is either negative or larger than number.of.variables then all
            elements of prior_inclusion_probabilities will be set to 1.0 and
            the model will be fit with all available coefficients.
          optional_coefficient_estimate: A vector of length number.of.variables
            to use as the prior mean of the regression coefficients.  This can
            also be None, in which case the prior mean for the intercept will
            be set to mean.y, and the prior mean for all slopes will be 0.
          max_flips: The maximum number of coefficients to investigate (in
            random order) each iteration.
          mean_y: The mean of the response variable.  Used to create a sensible
            default prior mean for the regression coefficients when
            optional_coefficient_estimate is None.
          sdy: Used along with expected_r2 to create a prior guess at the
            residual variance.
          prior_inclusion_probabilities: A vector of length number.of.variables
            giving the prior inclusion probability of each coefficient.  Each
            element must be between 0 and 1, inclusive.  If left as None then a
            default value will be created with all elements set to
            expected_model_size / number_of_variables.
          sigma_upper_limit: The largest acceptable value for the residual
            standard deviation.
          max_size: Assign prior probabilty zero to models with more than this
            many nonzero coefficients.
        """

        if isinstance(x, RegressionSuf):
            (xtx, mean_y_data, sdy_data,
             sample_size) = self._init_from_suf(x)
        else:
            (xtx, mean_y_data, sdy_data,
             sample_size) = self._init_from_data(x, y)

        xdim = xtx.shape[0]
        if mean_y is None:
            mean_y = mean_y_data
        if sdy is None:
            sdy = sdy_data

        if optional_coefficient_estimate is None:
            optional_coefficient_estimate = np.zeros(xdim)
            optional_coefficient_estimate[0] = mean_y
        self._mean = boom.Vector(optional_coefficient_estimate)

        D = np.diag(np.diagonal(xtx))
        ominv = diagonal_shrinkage * D + (1 - diagonal_shrinkage) * xtx
        ominv *= prior_information_weight / sample_size

        self._unscaled_prior_precision = to_boom_spd(ominv)

        if prior_inclusion_probabilities is None:
            prob = expected_model_size / xdim
            if prob > 1:
                prob = 1
            if prob < 0:
                prob = 0
            self._prior_inclusion_probabilities = boom.Vector(
                np.full(xdim, prob))
        else:
            self._prior_inclusion_probabilities = boom.Vector(
                prior_inclusion_probabilities)

        sample_variance = sdy**2
        expected_residual_variance = (1 - expected_r2) * sample_variance
        self._residual_precision_prior = boom.ChisqModel(
            prior_df,
            np.sqrt(expected_residual_variance))

        if sigma_upper_limit is None:
            self._sigma_upper_limit = sdy * 1.2
        else:
            self._sigma_upper_limit = sigma_upper_limit

        self._max_flips = max_flips

        self._max_size = max_size

    def slab(self, boom_sigsq_prm):
        """Return a boom.MvnGivenScalarSigma model.

        Args:
          boom_sigsq_prm: A BOOM::Ptr<UnivParams> to the residual variance
            parameter for the regression model.

        Returns:
          A BOOM::Ptr<MvnGivenScalarSigma> model that can serve as the slab in
          a spike and slab regression model.

        """
        return boom.MvnGivenScalarSigma(
            self._mean, self._unscaled_prior_precision, boom_sigsq_prm)

    @property
    def spike(self):
        ans = boom.VariableSelectionPrior(self._prior_inclusion_probabilities)
        if (
                (self.max_size is not None)
                and (self.max_size > 0)
                and np.isfinite(self.max_size)
        ):
            ans.set_max_size(int(self.max_size))

        return ans

    @property
    def residual_precision(self):
        """
        A boom.ChisqModel giving the prior distribution for the residual
        precision parameter.
        """
        return self._residual_precision_prior

    @property
    def sigma_upper_limit(self):
        return self._sigma_upper_limit

    @property
    def max_flips(self):
        return self._max_flips

    @property
    def max_size(self):
        """
        Models with more than this many nonzero coefficients are assigned
        zero prior probability.  If there is no max size then max_size is None.
        """
        return self._max_size

    def create_sampler(self, model):
        """
        Args:
          model:  A boom.RegressionModel object.
        
        Returns:
          A boom.BregVsSampler object.
        """
        sampler = boom.BregVsSampler(model,
                                     self.slab(model.Sigsq_prm),
                                     self.residual_precision,
                                     self.spike)
        if ((self.max_flips is not None)
            and (self.max_flips > 0)
            and (np.isfinite(self.max_flips))):
            sampler.limit_model_selection(self.max_flips)

        return sampler
                                          
    def _init_from_data(self, x, y):
        x = np.array(x)
        xtx = x.T @ x
        sample_size = x.shape[0]
        if y is None:
            mean_y = None
            sdy = None
        else:
            y = to_boom_vector(y)
            mean_y = boom.mean(y)
            sdy = boom.sd(y)

        return xtx, mean_y, sdy, sample_size

    def _init_from_suf(self, suf):
        return (
            suf.xtx,
            suf.mean_y,
            suf.sample_sd,
            suf.sample_size
        )
    
    def __getstate__(self):
        """
        Allows objects to be pickled.
        """
        ans = self.__dict__.copy()
        if hasattr(self, "_residual_precision_prior"):
            prior = self._residual_precision_prior
            ans["prior_df"] = 2.0 * prior.alpha
            ans["prior_ss"] = 2.0 * prior.beta
        del ans["_residual_precision_prior"]
        return ans

    def __setstate__(self, payload):
        """
        Allows objects to be unpickled.
        """
        self.__dict__.update(payload)
        self._residual_precision_prior = boom.ChisqModel(
            self.prior_df, np.sqrt(self.prior_ss / self.prior_df))
        del self.prior_df
        del self.prior_ss



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
                 suf: RegressionSuf,
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
    
        
