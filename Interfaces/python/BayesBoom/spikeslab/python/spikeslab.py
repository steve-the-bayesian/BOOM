import numpy as np
import pandas as pd
import patsy
import BayesBoom as boom

class SpikeSlabPriorBase:
    """Components of spike and slab priors that are shared regardless of the model
    type.

    """
    def __init__(self):
        """Computes information that is shared by the different implementation of spike
        and slab priors.  Currently, the only difference between the different
        priors is the prior variance on the regression coefficients.  When that
        changes, change this class accordingly, and change all the classes that
        inherit from it.

        Args:
          number_of_variables: The number of columns in the design matrix for
            the regression begin modeled.  The maximum size of the coefficient
            vector.

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
            also be None, in which case the prior mean for the intercept will be
            set to mean.y, and the prior mean for all slopes will be 0.

          mean.y: The mean of the response variable.  Used to create a sensible
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

        """
        pass


class SpikeSlabPrior(SpikeSlabPriorBase):
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
                 sigma_upper_limit=np.Inf):
        assert isinstance(x, np.ndarray)
        assert len(x.shape) == 2

        self._uscaled_prior_precsion = x.T @ x
        if mean_y is None:
            if y is None:
                throw Exception("Either y or mean_y must be specified.")
            mean_y = np.mean(y)
        if optional_coefficient_estimate is None:
            optional_coefficient_estimate = np.zeros(x.shape[1])
        pass


class lm_spike:
    """Fit a linear model with a spike and slab prior using MCMC.
    """

    def __init__(self,
                 formula: str,
                 niter: int,
                 data: DataFrame,
                 prior: SpikeSlabPrior = None,
                 ping: int = None,
                 seed: int = None,
                 **kwargs):
        """Create and a model object and run a specified number of MCMC iterations.

        Args:
          formula: A model formula that can be interpreted by the 'patsy' module
            to produce a model matrix from 'data'.
          niter: The desired number of MCMC iterations.
          data: A pd.DataFrame containing the data with which to train the
            model.
          prior: A SpikeSlabPrior object providing the prior distribution over
            the inclusion indicators, the coefficients, and the residual
            variance parameter.
          ping: The frequency (in iterations) with which to print status
            updates.  If ping is None then niter/10 will be assumed.
          seed: The seed for the C++ random number generator, or None.
          **kwargs: Extra argumnts will be passed to SpikeSlabPrior.

        Returns:
          An lm_spike object.

        """

        response, predictors = patsy.dmatrices(formula, data)
        xdim = predictors.shape[1]
        sample_size = predictors.shape[0]
        assert isinstance(niter, int)
        assert int > 0
        if ping is None:
            ping = niter / 10
        assert isinstance(ping, int)

        if seed is not None:
            assert isinstance(seed, int)
            boom.GlobalRng.rng.seed(seed)

        self._model = boom.RegressionModel(boom.Matrix(predictors),
                                           boom.Vector(response),
                                           False)
        sampler = boom.BregVsSampler(
            self._model, prior.slab, prior.residual_precision, prior.spike)
        self._model.set_method(sampler)

        self._coefficient_draws = []
        self._inclusion = []
        self._residual_sd = np.zeros(niter)
        self._log_likelihood = np.zeros(niter)

        for i in range(niter):
            if ping > 0:
                boom.print_timestamp(i, ping)
            self._model.sample_posterior()
            self._residual_sd = self._model.sigma()
            beta = self._model.coef()
            self._inclusion.append(
                np.array(beta.inc().included_positions().copy()))
            self._coefficient_draws.append(
                beta.included_coefficients())

    def plot(self, what=None):
        pass

    def predict(self, newdata, burn=None, seed=None):
        pass

    def suggest_burn(self):
        pass

    def summary(self, burn=None):
        pass

    def residuals(self, burn=None):
        pass
