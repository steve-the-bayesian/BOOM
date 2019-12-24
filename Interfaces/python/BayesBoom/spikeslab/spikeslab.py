import numpy as np
import pandas as pd
import patsy
import BayesBoom as boom


class RegressionSpikeSlabPrior:
    """Components of spike and slab priors that are shared regardless of the model
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
                 sigma_upper_limit=np.Inf):
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
        if isinstance(x, np.ndarray):
            x = boom.Matrix(x)
        assert isinstance(x, boom.Matrix)

        if mean_y is None:
            if y is None:
                raise Exception("Either 'y' or 'mean_y' must be specified.")
            if isinstance(y, np.ndarray):
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
                potential_nvars, prob)
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

    def slab(self, boom_sigsq_prm):
        """Return a BayesBoom.MvnGivenScalarSigma model.

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
        return self._residual_precision_prior


class lm_spike:
    """Fit a linear model with a spike and slab prior using MCMC.

    Typical use:

    from BayesBoom.spikeslab import lm_spike
    from sklearn.model_selection import train_test_split

    data = pd.read_csv("mydata.csv")
    train_data, pred_data = train_test_split(data, test_size=100)

    model = lm_spike('y ~ .', niter=1000, data=train_data)
    pred = model.predict(pred_data)

    model.plot()
    model.plot("coefficients")
    model.plot("inc")
    model.plot("resid")

    model.summary()
    model.residuals()

    pred.plot()

    """

    def __init__(self,
                 formula: str,
                 niter: int,
                 data: pd.DataFrame,
                 prior: RegressionSpikeSlabPrior = None,
                 ping: int = None,
                 seed: int = None,
                 **kwargs):
        """Create and a model object and run a specified number of MCMC iterations.

        Args:
          formula: A model formula that can be interpreted by the 'patsy'
            module to produce a model matrix from 'data'.
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
        # xdim = predictors.shape[1]
        # sample_size = predictors.shape[0]
        assert isinstance(niter, int)
        assert niter > 0
        if ping is None:
            ping = int(niter / 10)
        assert isinstance(ping, int)

        if seed is not None:
            assert isinstance(seed, int)
            boom.GlobalRng.rng.seed(seed)

        X = boom.Matrix(predictors)
        y = boom.Vector(response)

        self._model = boom.RegressionModel(X, y, False)
        if prior is None:
            prior = RegressionSpikeSlabPrior(x=X, y=y, **kwargs)

        sampler = boom.BregVsSampler(
            self._model,
            prior.slab(self._model.Sigsq_prm),
            prior.residual_precision,
            prior.spike)
        import pdb
        pdb.set_trace()
        self._model.set_method(sampler)
        self._coefficient_draws = []
        self._inclusion = []
        self._residual_sd = np.zeros(niter)
        self._log_likelihood = np.zeros(niter)

        for i in range(niter):
            self._model.sample_posterior()
            self._residual_sd = self._model.sigma()
            beta = self._model.coef
            self._inclusion.append(
                np.array(beta.inc().included_positions().copy()))
            self._coefficient_draws.append(
                beta.included_coefficients())
            self._log_likelihood[i] = self._model.log_likelihood()

    def plot(self, what=None, **kwargs):
        plot_types = [
            "coefficients",
            "inclusion"
            ]

    def predict(self, newdata, burn=None, seed=None):
        """
        Return an LmSpikePrediciton object.
        """

    def suggest_burn(self):
        pass

    def summary(self, burn=None):
        pass

    def residuals(self, burn=None):
        pass
