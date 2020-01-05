import BayesBoom as boom
import numpy as np
import patsy
from abc import ABC, abstractmethod
import spikeslab
import R
import scipy.sparse


# ===========================================================================
class StateModel(ABC):
    """StateModel objects are wrappers around boom.StateModel objects, which are
    opaquie C++ types.  The primary purpose of a separate python object is to
    handle various accounting duties required of StateModel's by the Bsts model
    defined below.

    """

    def __init__(self, state_index_begin):
        """
        Args:
          state_index_begin: The index of the state subcomponent described by
            this state model, in the full state vector.

        """
        self._state_index = state_index_begin

    @property
    @abstractmethod
    def state_dimension(self):
        """The dimension of the state subcomponent managed by this model.
        """

    @abstractmethod
    def allocate_space(self, niter):
        """Allocate the space needed to call 'record_state' the given number of times.

        Args:
          niter:  The number of iterations worth of space to be allocated.
        """

    @abstractmethod
    def record_state(self, iteration, state_matrix):
        """Record the state of any model parameters, and the subset of the state
        vector associated with this model, so they can be analyzed later.

        Args:
          iteration:  The (integer) index of the MCMC iteration to be recorded.
          state_matrix: The current matrix containing state.  Each column is a
            state vector associated with the corresponding time point.

        """


# ---------------------------------------------------------------------------
class LocalLevelModel(StateModel):
    """ The local level model assumes the data move as a noisy random walk.
              y[t] = mu[t] + error[t],
           mu[t+1] = mu[t] + innovation[t]    innovation[t] ~ N(0, sigsq)

    The model parameter is the variance of the innovation terms.
    """

    def __init__(self, state_index_begin, y, sigma_prior=None,
                 initial_state_prior=None, sdy=None, initial_y=None):
        """x
        Args:

          y: The data to be modeled.  If sdy and initial_y are supplied this is
            not used.

          sigma_prior: An object of class boom.GammaModelBase serving as the
            prior on the precision (reciprocal variance) of the innovation
            terms.  If None then 'sdy' will be used to choose a defalt.

          initial_state_prior: An object of class boom.GaussianModel serving as
            the prior distribution on the value of the state at time 0 (the
            time of the first observation).  If None then initial_y and sdy
            will be used to choose a defalt.

          sdy: The standard deviation of y.  If None then this will be computed
            from y.  This argument is primarily intended to handle unusual
            cases where 'y' is unavailable.

          initial_y: The first element of y.  If None then this will be
            computed from y.  This argument is primarily intended to handle
            unusual cases where 'y' is unavailable.

        Returns:
          A StateModel object representing a local level model.

        """
        super().__init__(state_index_begin)

        if sigma_prior is None:
            if sdy is None:
                sdy = np.std(y)
            sigma_prior = R.SdPrior(sigma_guess=.01 * sdy,
                                    sample_size=.01,
                                    upper_limit=sdy)
        assert(isinstance(sigma_prior, R.SdPrior))

        if initial_state_prior is None:
            if initial_y is None:
                initial_y = y[0]
            if sdy is None:
                sdy = np.std(y)
            assert isinstance(initial_y, float)
            assert isinstance(sdy, float)
            initial_state_prior = boom.GaussianModel(initial_y, sdy**2)
        assert(isinstance(initial_state_prior, boom.GaussianModel))

        self._state_model = boom.LocalLevelStateModel()
        self._state_model.set_initial_state_mean(initial_state_prior.mu)
        self._state_model.set_initial_state_variance(
            initial_state_prior.sigsq)

        innovation_precision_prior = boom.ChisqModel(
            sigma_prior.sigma_guess,
            sigma_prior.sample_size)
        state_model_sampler = boom.ZeroMeanGaussianConjSampler(
            self._state_model,
            innovation_precision_prior,
            seeding_rng=boom.GlobalRng.rng)
        state_model_sampler.set_upper_limit(sigma_prior.upper_limit)
        self._state_model.set_method(state_model_sampler)

    def __repr__(self):
        return f"Local level with sigma = {self._state_model.sigma}"

    @property
    def state_dimension(self):
        return 1

    def allocate_space(self, niter, time_dimension):
        self.sigma_draws = np.zeros(niter)
        self.state_contribution = np.zeros(niter, time_dimension)

    def record_state(self, i, state_matrix):
        self.sigma_draws[i] = self._state_model.sigma
        self.state_contribution[i, :] = state_matrix[self._state_index, :]


# ===========================================================================
class SeasonalStateModel(StateModel):
    """ Seasonal state model.  TODO(steve): build this.
    """


# ===========================================================================
class Bsts:
    """A Bayesian structural time series model.

    """

    def __init__(self, family="gaussian", seed=None):
        self._family = R.unique_match(family, ["gaussian", "poisson",
                                               "binomial", "student"])
        self._model = None
        self._state_models = []

    def add_local_level(self):
        pass

    def add_local_linear_trend(self):
        pass

    def add_seasonal(self):
        pass

    def train(self, formula, data, niter: int, prior=None, ping: int = None):
        """Train a bsts model by running a specified number of MCMC iterations.

        Args:
          formula: Either numeric time series (array-like), or a string giving
            a formula that can be interpreted by the 'patsy' package (python's
            version of R's model syntax).
          data: If a formula is given, data is a DataFrame containing the
            variables from the formula.  If 'formula' is a numeric then 'data'
            need not be specified.
          prior: The prior distribution for the observation model.  If a
            regression component is included then this is a
            spikeslab.RegressionSpikeSlabPrior describing the regression
            coefficients and the residual standard deviation.  Otherwise it is
            a boom.SdPrior on the residual standard deviation.  If None then a
            default prior will be chosen.
          niter:  The desired number of MCMC iterations.
          ping: The frequency with which to print status updates in the MCMC
            algorithm.  The default is niter/10.  If ping <= 0 then no status
            updates are printed.

        Effects:
          The model is populated with MCMC draws.  The regression parameters,
          if any, are stored in the model object itself.  Parameters for any
          state models are stored in the state model objects.  The
          contributions of each state model are stored in the state model
          objects.

        """
        self._create_model(formula, data, prior)
        self._allocate_space(niter, self.time_dimension)

        for i in range(niter):
            self._model.sample_posterior()
            self._record_state(i)

    def _record_state(self, i):
        """Record the state from the
        """
        state_matrix = self._model.state()
        for m in self._state_models:
            m.record_state(i, state_matrix)

        if self._have_regression:
            beta = self._model.coef
            self._coefficients[i, :] = spikeslab.lm_spike.sparsify(beta)

        self._residual_sd[i] = self._model.sigma

    def _format_data(self, formula, data):
        """
        """

    def _allocate_space(self, niter):
        for state_model in self._state_models:
            state_model._allocate_space(niter, self.time_dimension)
        if self._have_regression:
            self._coefficients = scipy.sparse.lil_matrix((niter, self.xdim))
        self._residual_sd = np.zeros(niter)

    def _create_model(self, formula, data, prior):
        """Create the boom model object.

        Args:
          formula: Either numeric time series (array-like), or a string giving
            a formula that can be interpreted by the 'patsy' package (python's
            version of R's model syntax).
          data: If a formula is given, data is a DataFrame containing the
            variables from the formula.  If 'formula' is a numeric then 'data'
            need not be specified.
          prior: The prior distribution for the observation model.  If a
            regression component is included then this is a
            spikeslab.RegressionSpikeSlabPrior describing the regression
            coefficients and the residual standard deviation.  Otherwise it is
            a boom.SdPrior on the residual standard deviation.  If None then a
            default prior will be chosen.

        Effects:
          self._model is created, populated with data and assigned a posterior
            sampler.
        """
        if isinstance(formula, str):
            self._create_state_space_regression_model(formula, data, prior)
        else:
            self._create_state_space_model(formula, prior)

    def _create_state_space_model(self, data, prior):
        """Create the boom model object.

        Args:
          data: The time series on which to base the model.
          prior: A boom.SdPior describing the prior distribution on the
            residual standard deviation.

        Effects:
          self._model is created, populated with data and assigned a posterior
            sampler.

        """
        assert isinstance(prior, boom.SdPior)
        time_series = boom.Vector(data)
        is_observed = np.isnan(time_series)
        self._model = boom.StateSpaceModel(time_series, is_observed)
        sampler = boom.StateSpacePosteriorSampler(
            self._model, prior, boom.GlobalRng.rng)
        self._model.set_method(sampler)


    def _create_state_space_regression_model(self, formula, data, prior):
        """Create the boom model object, and store related model artifacts.

        Args:
          formula:  A model formula describing the regression component.
          data: A pandas DataFrame containing the variables appearing
            'formula'.
          prior: A spikeslab.RegressionSpikeSlabPrior describing the prior
            distribution on the regression coefficients and the residual
            standard deviation.

        Effects: self._model is created, and model formula artifacts are stored
          so they will be available for future predictions.

        """
        assert isinstance(prior, spikeslab.RegressionSpikeSlabPrior)
        assert isinstance(formula, str)
        response, predictors = patsy.dmatrices(formula, data)
        is_observed = np.isnan(response)

        self._model = boom.StateSpaceRegressionModel(
            boom.Vector(response),
            boom.Matrix(predictors),
            is_observed)

        ## handle upper_limit

        spikeslab.set_posterior_sampler(self._model.observation_model, prior)
