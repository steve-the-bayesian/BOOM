import BayesBoom as boom
import numpy as np
import patsy
import spikeslab
import R
import scipy.sparse

from state_models import StateModel


class Bsts:
    """A Bayesian structural time series model.

    """

    def __init__(self, family="gaussian", seed=None):
        self._family = R.unique_match(family, ["gaussian", "poisson",
                                               "binomial", "student"])
        self._model = None
        self._state_models = []
        self._state_dimension = 0

    def add_state(self, state_model: StateModel):
        """Add a component of state to the model.

        Args:
          state_model: A StateModel object describing the state to be added.

        Effects:
          The state model is appended to self._state_models, and the state model
          is informed about the position of the state it manages in the global
          state vector.

        """
        state_model.set_state_index(self._state_dimension)
        self._state_dimension += state_model.state_dimension
        self._state_models.append(state_model)

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
        for state_model in self._state_models:
            self._model.add_state(state_model)
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
        for state_model in self._state_models:
            self._model.add_state(state_model)
