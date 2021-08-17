import BayesBoom.boom as boom
import numpy as np
import patsy
import BayesBoom.R as R
from numbers import Number
# import BayesBoom.spikeslab as spikeslab
# import scipy.sparse


class SparseDynamicRegressionModel:
    """
    A DynamicRegressionModel is a time series regression model where the
    coefficients obey a classic state space model.  Note that the number of
    observations at each time point might differ.  The model is implemented as
    a multivariate state space model.  Through data augmentation one can extend
    this model to most GLM's.

    Define the set of responses at time t as Y'_t = [y_1t, y_2t, ... y_n_tt],
    where Y_t = X[t] * beta[t] + error[t], with temporally IID error term
    error[t] ~ N(0, Diagonal(sigma^2)).

    Each coefficient beta[j, t] is zero with probability determined by a Markov
    chain Pr(beta[j, t] = s | beta[j, t-1] = r) = q[r, s], for r, s in {0, 1}.

    The conditional distribution of beta[j, t] given beta[j, t-1], and given
    that both are nonzero, is normal with mean b_jt = T_ij b_jt-1, and variance
    tau^2.

    Expected Use:

    data = get_data_from_somewhere()
    model = SparseDynamicRegressionModel(
        y ~ dot(data, ["y", "timestamp_column"]),
        data=data,
        timestamps="timestamp_column",
        niter=1000,
    )

    """

    def __init__(self,
                 formula,
                 data,
                 timestamps,
                 niter: int = 0,
                 residual_precision_prior=None,
                 coefficient_innovation_priors=None,
                 prior_inclusion_probabilities=None,
                 expected_inclusion_duration=None,
                 transition_probability_prior_sample_size=None,
                 ping=None,
                 seed=None):
        """
        Args:
          formula: A string of the form "response ~ v1 + v2" where "y", "v1"
            and "v2" are named columns in 'data'.  See the 'dot' function in
            the BayesBoom.spikeslab package for specifying complex formulas
            simply.
          data: A pd.DataFrame or equivalent containing the data used in the
            model.
          timestamps: Either a vector of type datetime64[ns] (Is this overly
            restrictive?) containing

          innovation_precision_priors:  Prior distribution on the residual
            precision parameter 1/sigsq.  This must be an object inheriting
            from boom.GammaModelBase.  The typcial choice is a boom.ChisqModel.


        """
        if isinstance(timestamps, str):
            self._timestamps = data[timestamps]
        else:
            self._timestamps = timestamps

        self._formula = formula
        self._data = data
        self._data_by_timestamp = None

        self._residual_precision_prior = residual_precision_prior
        self._coefficient_innovation_priors = coefficient_innovation_priors
        self._prior_inclusion_probabilities = prior_inclusion_probabilities
        self._expected_inclusion_duration = expected_inclusion_duration
        tmp = transition_probability_prior_sample_size
        self._transition_probability_prior_sample_size = tmp

        self._model = None

        if niter > 0:
            self.train(niter=niter, ping=ping, seed=seed)

    @property
    def xdim(self):
        """
        The number of potential predictor variables.
        """
        if self._model:
            return self._model.xdim
        elif self._data is not None and self._formula is not None:
            if "~" in self._formula:
                pred_formula = self._formula.split("~")[1]
            else:
                pred_formula = self._formula
            predictors = patsy.dmatrix(pred_formula, self._data.iloc[:2, :])
            return predictors.shape[1]
        else:
            return 0

    @property
    def time_dimension(self):
        """
        The number of time points in the model's data.
        """
        if self._model:
            return self._model.time_dimension
        return 0

    def train(self, niter: int, ping: int = None, seed: int = None):
        self._create_model()
        self._set_data(self._formula, self._data, self._timestamps)
        self._validate_priors()
        self._create_posterior_sampler(
            self._residual_precision_prior,
            self._coefficient_innovation_priors,
            self._prior_inclusion_probabilities,
            self._expected_inclusion_duration,
            self._transition_probability_prior_sample_size)

        if seed is not None:
            np.random.seed(seed)
            boom.GlobalRng.rng.seed(seed)
        self._allocate_space(niter)
        for i in range(niter):
            self._model.sample_posterior()
            self._record_state(i)

    def _create_model(self):
        """
        Populate self._model with the Boom model object.
        """
        self._model = boom.DynamicRegressionModel(self.xdim)

    def _set_data(self, formula, data, timestamps):
        """
        Partitiion the DataFrame 'data' into chunks defined by 'timestamps',
        pass it through the 'formula', and convert the output to the
        expected BayesBoom objects.

        Args:
          formula: A string defining a formula as interpreted by the 'patsy'
            library.
          data: A data frame containing the variables in 'formula', and
            maybe other variables as well.  Extraneous variables will be
            ignored.
          timestamps: A vector-like object containing objects that can be
            ordered.  E.g. a vector of dates, or integers.  Each element of
            'timestamps' corresponds to a row of 'data', and determines
            which the time point to which that row belongs.

        Effects:
          * Data are added to the BOOM model.
          * self._response_suf is created and populated with response data.
          * self._predictor_suf is created and each element is populated with
              data from the corresponding predictor variable.
        """

        unique_time_points = sorted(set(timestamps))
        self._response_suf = R.GaussianSuf()
        xdim = self.xdim
        self._predictor_suf = [R.GaussianSuf()] * xdim
        for time_stamp in unique_time_points:
            subset = timestamps == time_stamp
            response, predictors = patsy.dmatrices(
                formula, data.loc[subset, :], eval_env=1)
            data_point = boom.RegressionDataTimePoint(
                boom.Matrix(predictors),
                boom.Vector(response))
            self._response_suf += response
            for i in range(xdim):
                self._predictor_suf[i] += predictors[:, i]

            self._model.add_data(data_point)

    def _validate_priors(self):
        self._validate_residual_precision_prior()
        self._validate_coefficient_innovation_priors()
        self._validate_inclusion_prior()

    def _validate_residual_precision_prior(self):
        """
        Ensure the self._residual_precision_prior is the correct class.

        Preconditions:
            self._set_data must have been run.
        """
        if isinstance(self._residual_precision_prior, R.SdPrior):
            return

        # Assume an expected R^2 of 50%.
        target_variance = self._response_suf.sample_variance / 2.0
        self._residual_precision_prior = R.SdPrior(
            sigma_guess=np.sqrt(target_variance),
            sample_size=1.0)

    def _validate_coefficient_innovation_priors(self):
        """
        Ensure that self._coefficient_innovation_priors are a list of SdPriors.
        """
        if (isinstance(self._coefficient_innovation_priors, list) and np.all(
                [isinstance(x, R.SdPrior) for x in
                 self._coefficient_innovation_priors])):
            return

        if isinstance(self._coefficient_innovation_priors, R.SdPrior):
            self._coefficient_innovation_priors = [
                self._coefficient_innovation_priors] * self.xdim
            return

        if self._coefficient_innovation_priors is not None:
            raise Exception("coefficient_innovation_priors must either be an "
                            "R.SdPrior or a list of such priors.")

        sdy = self._response_suf.sample_sd
        self._coefficient_innovation_priors = [
            R.SdPrior(.01 * sdy / self._predictor_suf[i].sample_sd, 1)
            for i in range(self.xdim)
        ]

    def _validate_inclusion_prior(self):
        xdim = self.xdim
        if self._expected_inclusion_duration is None:
            self._expected_inclusion_duration = [50] * xdim
        elif isinstance(self._expected_inclusion_duration, Number):
            self._expected_inclusion_duration = [
                self._expected_inclusion_duration] * xdim
        if len(self._expected_inclusion_duration) != xdim:
            raise Exception(f"Expected {xdim} numbers in expected"
                            "_inclusion_duration.")
        elif np.any([x <= 0 for x in self._expected_inclusion_duration]):
            raise Exception("All entries in expected_inclusion_duration"
                            " must be positive")
        self._expected_inclusion_duration = np.array(
            self._expected_inclusion_duration, dtype=float)

        if self._prior_inclusion_probabilities is None:
            self._prior_inclusion_probabilities = [1.0 / xdim] * xdim
        elif isinstance(self._prior_inclusion_probabilities, Number):
            self._prior_inclusion_probabilities = [
                self._prior_inclusion_probabilities] * xdim
        if len(self._prior_inclusion_probabilities) != xdim:
            raise Exception(f"Expected {xdim} elements in prior_inclusion_"
                            "probabilities.")
        elif np.any([x < 0 for x in self._prior_inclusion_probabilities]):
            raise Exception("All prior_inclusion_probabilities must be "
                            "non-negative.")
        elif np.any([x > 1 for x in self._prior_inclusion_probabilities]):
            raise Exception("All prior_inclusion_probabilities must be less"
                            "than 1.")
        self._prior_inclusion_probabilities = np.array(
            self._prior_inclusion_probabilities, dtype=float)

        if self._transition_probability_prior_sample_size is None:
            self._transition_probability_prior_sample_size = [1.0] * xdim
        elif isinstance(self._transition_probability_prior_sample_size,
                        Number):
            self._transition_probability_prior_sample_size = [
                self._transition_probability_prior_sample_size] * xdim

        if len(self._transition_probability_prior_sample_size) != xdim:
            raise Exception(f"Expected {xdim} elements in transition_"
                            "probability_prior_sample_size.")
        elif np.any([x <= 0 for x in
                     self._transition_probability_prior_sample_size]):
            raise Exception("All elements of transition_probability_prior_"
                            "sample_size must be positive.")
        self._transition_probability_prior_sample_size = np.array(
            self._transition_probability_prior_sample_size, dtype=float)

    def _create_posterior_sampler(
            self,
            residual_precision_prior,
            coefficient_innovation_priors,
            prior_inclusion_probabilities,
            expected_inclusion_duration,
            transition_probability_prior_sample_size):
        sampler = boom.DynamicRegressionDirectGibbsSampler(
            self._model,
            residual_precision_prior.sigma_guess,
            residual_precision_prior.sample_size,
            boom.Vector(np.array(
                [x.sigma_guess for x in coefficient_innovation_priors])),
            boom.Vector(np.array(
                [x.sample_size for x in coefficient_innovation_priors])),
            boom.Vector(prior_inclusion_probabilities),
            boom.Vector(expected_inclusion_duration),
            boom.Vector(transition_probability_prior_sample_size))
        self._model.set_method(sampler)

    def _allocate_space(self, niter):
        """
        Create space to store 'niter' MCMC draws.
        """
        self._beta_draws = np.zeros((niter, self.xdim, self.time_dimension))
        self._residual_sd_draws = np.zeros((niter))
        self._innovation_sd_draws = np.zeros((niter, self.xdim))
        self._transition_probabilities = np.zeros((niter, self.xdim, 2, 2))

    def _record_state(self, i):
        # TODO: all_coefficients
        self._beta_draws[i, :, :] = self._model.all_coefficients.to_numpy()
        self._residual_sd_draws[i] = self._model.residual_sd
        self._innovation_sd_draws[i, :] = (
            self._model.unscaled_innovation_sds
            * self._model.residual_sd).to_numpy()
        for pred in range(self.xdim):
            self._transition_probabilities[i, pred, :, :] = (
                self._model.transition_probabilities(pred).to_numpy()
            )
