import numpy as np
import pandas as pd
import BayesBoom.boom as boom
import BayesBoom.R as R


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
                 sigma_upper_limit=None):
        """
        Computes information that is shared by the different implementation of
        spike and slab priors.  Currently, the only difference between the
        different priors is the prior variance on the regression coefficients.
        When that changes, change this class accordingly, and change all the
        classes that inherit from it.

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
            also be None, in which case the prior mean for the intercept will
            be set to mean.y, and the prior mean for all slopes will be 0.

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
        if isinstance(x, pd.DataFrame):
            all_numeric = np.all(x.dtypes.apply(pd.api.types.is_numeric_dtype))
            if all_numeric:
                x = boom.Matrix(x.values)
            else:
                raise Exception("A data frame that included non-numeric data "
                                "was passed to RegressionSpikeSlabPrior.")
        elif isinstance(x, np.ndarray):
            x = boom.Matrix(x)
        if not isinstance(x, boom.Matrix):
            raise Exception(
                "x should either be a 2-dimensional np.array or a boom.Matrix.")

        if mean_y is None:
            if y is None:
                raise Exception("Either 'y' or 'mean_y' must be specified.")
            if isinstance(y, pd.Series):
                y = boom.Vector(y.values)
            elif isinstance(y, np.ndarray):
                y = boom.Vector(y)
            mean_y = boom.mean(y)
        if optional_coefficient_estimate is None:
            optional_coefficient_estimate = np.zeros(x.ncol)
            optional_coefficient_estimate[0] = mean_y
        self._mean = boom.Vector(optional_coefficient_estimate)

        sample_size = x.nrow
        ods = 1. - diagonal_shrinkage
        scale_factor = prior_information_weight * ods / sample_size
        self._unscaled_prior_precision = x.inner() * scale_factor
        diag_view = self._unscaled_prior_precision.diag()
        diag_view /= ods

        if prior_inclusion_probabilities is None:
            potential_nvars = x.ncol
            prob = expected_model_size / potential_nvars
            if prob > 1:
                prob = 1
            if prob < 0:
                prob = 0
            self._prior_inclusion_probabilities = boom.Vector(
                np.full(potential_nvars, prob))
        else:
            self._prior_inclusion_probabilities = boom.Vector(
                prior_inclusion_probabilities)

        if sdy is None:
            sdy = boom.sd(y)
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

    def __getstate__(self):
        ans = self.__dict__.copy()
        if hasattr(self, "_residual_precision_prior"):
            prior = self._residual_precision_prior
            ans["prior_df"] = 2.0 * prior.alpha()
            ans["prior_ss"] = 2.0 * prior.beta()
        del ans["_residual_precision_prior"]
        return ans

    def __setstate__(self, payload):
        self.__dict__.update(payload)
        self._residual_precision_prior = boom.ChisqModel(
            self.prior_df, np.sqrt(self.prior_ss / self.prior_df))
        del self.prior_df
        del self.prior_ss

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
        return boom.VariableSelectionPrior(self._prior_inclusion_probabilities)

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


class StudentSpikeSlabPrior(RegressionSpikeSlabPrior):
    """
    A SpikeSlabPrior appropriate for regression models with Student T errors.
    """
    def __init__(
            self,
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
            sigma_upper_limit=np.Inf,
            tail_thickness_prior=R.UniformPrior(0.1, 100)
    ):
        """
        Args:
          tail_thickness_prior: An object with a boom() method, which returns a
            boom.DoubleModel describing the prior on the tail thickness
            parameter.

          All other arguments are as documented in RegressionSpikeSlabPrior.
        """
        super().__init__(
            x=x,
            y=y,
            expected_r2=expected_r2,
            prior_df=prior_df,
            expected_model_size=expected_model_size,
            prior_information_weight=prior_information_weight,
            diagonal_shrinkage=diagonal_shrinkage,
            optional_coefficient_estimate=optional_coefficient_estimate,
            max_flips=max_flips,
            mean_y=mean_y,
            sdy=sdy,
            prior_inclusion_probabilities=prior_inclusion_probabilities,
            sigma_upper_limit=sigma_upper_limit)
        self._nu_prior = tail_thickness_prior

    @property
    def tail_thickness(self):
        """
        A boom.DoubleModel giving the prior distribution on the tail thickness
        parameter.
        """
        return self._nu_prior.boom()

    def create_sampler(self, model, assign=False):
        pass


def logit(p):
    ans = np.log(p / (1 - p))
    return float(ans)


class LogitZellnerPrior:
    def __init__(self,
                 predictors,
                 successes=None,
                 trials=None,
                 prior_success_probability=0.5,
                 expected_model_size=1.0,
                 prior_information_weight=1.0,
                 diagonal_shrinkage=.5,
                 optional_coefficient_estimate=None,
                 max_flips=-1,
                 prior_inclusion_probabilities=None):
        """
        Args:
          predictors: A matrix of predictor variables, denoted X below.
          successes:  A vector of success counts.
          trials: A vector of trial counts.  If successes and trials are given,
            they must satisfy 0 <= successes <= trials.  If successes is given
            and trials is None, then successes must only contain 0's and 1's.

          prior_success_probability: If 'successes' is not given,
            then prior_success_probability is used to scale the prior mean of
            the intercept term.  Otherwise
        """

        self._max_flips = max_flips
        sample_size = predictors.shape[0]
        xdim = predictors.shape[1]

        xtx = predictors.T @ predictors * prior_information_weight / sample_size
        xtx_diagonal = np.diagonal(xtx).copy()
        xtx *= 1 - diagonal_shrinkage
        np.fill_diagonal(xtx, xtx_diagonal)

        self._precision = xtx
        self._mean = np.zeros(xdim)
        if successes is None:
            self._mean[0] = logit(prior_success_probability)
        else:
            if trials is None:
                trials = np.ones(sample_size)
            p_hat = np.nanmean(successes / trials)
            self._mean[0] = logit(p_hat)
        if not np.isfinite(self._mean[0]):
            self._mean[0] = 0.0

        if prior_inclusion_probabilities is None:
            prior_inclusion_probabilities = np.full(
                xdim, expected_model_size / xdim)
        self._prior_inclusion_probabilities = prior_inclusion_probabilities

    @classmethod
    def from_parameters(self, mean, precision, prior_inclusion_probabilities,
                        max_flips=-1):
        xdim = len(mean)
        predictors = np.random.randn(xdim, xdim)
        y = np.full(xdim, 0.5)
        trials = np.ones(xdim)
        ans = LogitZellnerPrior(predictors, y, trials, max_flips=max_flips)
        ans._prior_inclusion_probabilities = prior_inclusion_probabilities
        ans._mean = mean
        ans._precision = precision
        return ans

    @property
    def slab(self):
        return boom.MvnModel(
            boom.Vector(self._mean),
            boom.SpdMatrix(self._precision),
            True)

    @property
    def spike(self):
        return boom.VariableSelectionPrior(
            self._prior_inclusion_probabilities)

    @property
    def max_flips(self):
        return self._max_flips

    def create_sampler(self, model, assign=False):
        if not isinstance(model, boom.BinomialLogitModel):
            raise Exception("Expected 'model' to be a boom.BinomialLogitModel.")
        sampler = boom.BinomialLogitSpikeSlabSampler(
            model=model,
            slab=self.slab,
            spike=self.spike,
            clt_threshold=5,
            seeding_rng=boom.GlobalRng.rng)
        if self._max_flips > 0 and np.isfinite(self._max_flips):
            sampler.limit_model_selection(self._max_flips)
        if assign:
            model.set_method(sampler)
        return sampler


class PoissonZellnerPrior:
    def __init__(self,
                 predictors,
                 counts=None,
                 exposure=None,
                 prior_event_rate=1.0,
                 expected_model_size=1.0,
                 prior_information_weight=1.0,
                 diagonal_shrinkage=.5,
                 optional_coefficient_estimate=None,
                 max_flips=-1,
                 prior_inclusion_probabilities=None):
        self._max_flips = max_flips
        sample_size = predictors.shape[0]
        xdim = predictors.shape[1]

        xtx = predictors.T @ predictors * prior_information_weight / sample_size
        xtx_diagonal = np.diagonal(xtx).copy()
        xtx *= 1 - diagonal_shrinkage
        np.fill_diagonal(xtx, xtx_diagonal)

        self._precision = xtx
        self._mean = np.zeros(xdim)
        if counts is None:
            self._mean[0] = np.log(prior_event_rate)
        else:
            if exposure is None:
                exposure = np.ones(sample_size)
            p_hat = np.nanmean(counts / exposure)
            self._mean[0] = logit(p_hat)
        if not np.isfinite(self._mean[0]):
            self._mean[0] = 0.0

        if prior_inclusion_probabilities is None:
            prior_inclusion_probabilities = np.full(
                xdim, expected_model_size / xdim)
        self._prior_inclusion_probabilities = prior_inclusion_probabilities

    @classmethod
    def from_parameters(self, mean, precision, prior_inclusion_probabilities,
                        max_flips=-1):
        xdim = len(mean)
        predictors = np.random.randn(xdim, xdim)
        y = np.full(xdim, 0.5)
        trials = np.ones(xdim)
        ans = LogitZellnerPrior(predictors, y, trials, max_flips=max_flips)
        ans._prior_inclusion_probabilities = prior_inclusion_probabilities
        ans._mean = mean
        ans._precision = precision
        return ans

    @property
    def slab(self):
        return boom.MvnModel(
            boom.Vector(self._mean),
            boom.SpdMatrix(self._precision),
            True)

    @property
    def spike(self):
        return boom.VariableSelectionPrior(
            self._prior_inclusion_probabilities)

    @property
    def max_flips(self):
        return self._max_flips

    def create_sampler(self, model, assign=False):
        if not isinstance(model, boom.PoissonRegressionModel):
            raise Exception(
                "Expected 'model' to be a boom.PoissonRegressionModel.")
        sampler = boom.PoissonRegressionSpikeSlabSampler(
            model=model,
            slab=self.slab,
            spike=self.spike,
            clt_threshold=5,
            seeding_rng=boom.GlobalRng.rng)
        if self._max_flips > 0 and np.isfinite(self._max_flips):
            sampler.limit_model_selection(self._max_flips)
        if assign:
            model.set_method(sampler)
        return sampler
