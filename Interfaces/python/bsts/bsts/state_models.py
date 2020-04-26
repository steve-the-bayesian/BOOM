import BayesBoom as boom
import numpy as np
# import patsy
from abc import ABC, abstractmethod
# import spikeslab
import R
# import scipy.sparse


# ===========================================================================
class StateModel(ABC):
    """StateModel objects are wrappers around boom.StateModel objects, which are
    opaquie C++ types.  The primary purpose of a separate python object is to
    handle various accounting duties required of StateModel's by the Bsts model
    defined below.

    """

    def __init__(self):
        """
        Args:
          state_index_begin: The index of the state subcomponent described by
            this state model, in the full state vector.

        """
        self._state_index = -1

    @property
    def state_index(self):
        """
        The index of the this state model's first element, in the global state
        vector.  This starts off as -1.  When the state model is added to a
        bsts model, bsts will call set_state_index() to inform the state model
        of this value.
        """
        if self._state_index < 0:
            raise Exception("Each state model must be told where its state"
                            "component begins in the global state vector.  "
                            "Try calling set_state_index.")
        return self._state_index

    def set_state_index(self, state_index_begin: int):
        """
        Inform the state model of the first index in the larger state vector
        corresponding to the state component it is responsible for.

        Args:
           state_index_begin: The smallest index in the global state vector
             corresponding to the component of state that this model describes.
        """
        self._state_index = state_index_begin

    @property
    @abstractmethod
    def state_dimension(self):
        """
        The dimension of the state subcomponent managed by this model.
        """

    @abstractmethod
    def allocate_space(self, niter):
        """
        Allocate the space needed to call 'record_state' the given number of
        times.

        Args:
          niter:  The number of iterations worth of space to be allocated.
        """

    @abstractmethod
    def record_state(self, iteration, state_matrix):
        """
        Record the state of any model parameters, and the subset of the state
        vector associated with this model, so they can be analyzed later.

        Args:
          iteration:  The (integer) index of the MCMC iteration to be recorded.
          state_matrix: The current matrix containing state.  Each column is a
            state vector associated with the corresponding time point.
        """


# ---------------------------------------------------------------------------
class LocalLevelStateModel(StateModel):
    """
    The local level model assumes the data move as a noisy random walk.
              y[t] = mu[t] + error[t], mu[t+1] = mu[t] + innovation[t]
     innovation[t] ~ N(0, sigsq)

    The model parameter is the variance of the innovation terms.
    """

    def __init__(self, y, sigma_prior=None, initial_state_prior=None,
                 sdy=None, initial_y=None):
        """
        Args:
          y: The data to be modeled.  If sdy and initial_y are supplied
            this is not used.
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
        state_model_sampler = self._state_model.set_posterior_sampler(
            innovation_precision_prior)
        state_model_sampler.set_sigma_upper_limit(sigma_prior.upper_limit)

    def __repr__(self):
        return f"Local level with sigma = {self._state_model.sigma}"

    @property
    def state_dimension(self):
        return 1

    def allocate_space(self, niter, time_dimension):
        self.sigma_draws = np.zeros(niter)
        self.state_contribution = np.zeros((niter, time_dimension))

    def record_state(self, i, state_matrix):
        self.sigma_draws[i] = self._state_model.sigma
        if self._state_index < 0:
            raise Exception("Each state model must be told where its state"
                            "component begins in the global state vector.  "
                            "Try calling set_state_index.")
        self.state_contribution[i, :] = state_matrix[self._state_index, :]


# ===========================================================================
class SeasonalStateModel(StateModel):
    """
    Seasonal state model.  TODO(steve): build this.
    """

    def __init__(self,
                 y,
                 nseasons: int,
                 season_duration: int = 1,
                 initial_state_prior: boom.MvnModel = None,
                 innovation_sd_prior: R.SdPrior = None,
                 sdy: float = None):
        """
        Args:
          y: The time series being modeled.  This can be omitted if either (a)
            initial_state_prior and sdy and initial_y are passed, or (b) sdy
            and initial_y are passed.
          nseasons: The number of seasons in a cycle.
          season_duration:  The number of time periods each season.  See below.
          initial_state_prior: A multivariate normal distribution of dimension
            nseasons - 1.  This is a distribution on the seasonal value at time
            0 and on the nseasons-2 previous values.  If None is passed then a
            default prior will be assumed.
          innovation_sd_prior: Prior distribution on the standard deviation of
            the innovation terms.  If None, then a default prior will be
            assumed.
          sdy: The standard deviation of the time series being modeled.

        Details:

        """
        self._state_model = boom.SeasonalStateModel(
            nseasons=nseasons, season_duration=season_duration)

        if initial_state_prior is None:
            if sdy is None:
                if y is None:
                    raise Exception("One of 'y', 'sdy', or "
                                    "'initial_state_prior' must be supplied.")
                sdy = np.nanstd(y)
            initial_state_prior = self._default_initial_state_prior(sdy)
        if innovation_sd_prior is None:
            if sdy is None:
                if y is None:
                    raise Exception("One of 'y', 'sdy', or "
                                    "'innovation_sd_prior' must be supplied.")
                sdy = np.nanstd(y)
            innovation_sd_prior = self._default_sigma_prior(sdy)

        self._state_model.set_initial_state_mean(
            initial_state_prior.mu)
        self._state_model.set_initial_state_variance(
            initial_state_prior.Sigma)

        innovation_precision_prior = boom.ChisqModel(
            innovation_sd_prior.sigma_guess,
            innovation_sd_prior.sample_size)
        state_model_sampler = boom.ZeroMeanGaussianConjSampler(
            self._state_model,
            innovation_precision_prior,
            seeding_rng=boom.GlobalRng.rng)

        state_model_sampler.set_sigma_upper_limit(
            innovation_sd_prior.upper_limit)
        self._state_model.set_method(state_model_sampler)

    def __repr__(self):
        ans = f"A SeasonalStateModel with {self.nseasons} "
        ans += f"seasonas of duration {self.season_duration}, and "
        ans += f"residual sd {self._state_model.sigma}."
        return ans

    @property
    def nseasons(self):
        return self._state_model.nseasons

    @property
    def season_duration(self):
        return self._state_model.season_duration

    @property
    def state_dimension(self):
        return self.nseasons - 1

    def allocate_space(self, niter, time_dimension):
        self.sigma_draws = np.zeros(niter)
        self.state_contribution = np.zeros((niter, time_dimension))

    def record_state(self, i, state_matrix):
        self.sigma_draws[i] = self._state_model.sigma
        self.state_contribution[i, :] = state_matrix[self._state_index, :]

    def _default_sigma_prior(self, sdy):
        """
        The default prior to use for the innovation standard deviation.
        """
        return R.SdPrior(.01 * sdy, upper_limit=sdy)

    def _default_initial_state_prior(self, sdy):
        """
        The default prior to use for the initial state vector.
        """
        dim = self.nseasons - 1
        return boom.MvnModel(boom.Vector(dim, 0.0), boom.SpdMatrix(int(dim), sdy))

# ===========================================================================
