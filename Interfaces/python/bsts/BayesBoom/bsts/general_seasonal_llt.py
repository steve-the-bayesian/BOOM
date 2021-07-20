import BayesBoom.boom as boom
import numpy as np
import BayesBoom.R as R
from .state_models import StateModel


class GeneralSeasonalLLT(StateModel):
    """
    A seasonal state model that builds a local trend into the seasonal values.
    """

    def __init__(self,
                 y,
                 nseasons: int,
                 initial_state_prior=None,
                 level_precision_priors=None,
                 slope_precision_priors=None,
                 sdy: float = None):
        """
        Args:
          y: The time series to be modeled.  This can be "None" if 'sdy' is
            supplied.
          nseasons: The number of seasons in a cycle.
          initial_state_prior: An R.NormalPrior object describing the initial
            distribution of the state at time 0.  If None then a default prior
            will be assumed.
          level_precision_priors: A list of R.SdPrior objects describing the
            prior distribution on the innovation standard deviations for the
            level portion of the model.  There is one such prior for each
            season in the cycle.
          slope_precision_priors: A list of R.SdPrior objects describing the
            prior distribution on the innovation standard deviations for the
            slope portion of the model.  There is one such prior for each
            season in the cycle.
          sdy: The standard deviation of the time series to be modeled.  This
            is not needed if 'y' is supplied, or if all the prior distributions
            are explicity supplied.
        """

        self._nseasons = int(nseasons)
        if nseasons <= 1:
            raise Exception("Seasonal models require at least 2 seasons.")

        if initial_state_prior is None:
            if sdy is None:
                sdy = self._compute_sdy(y, "initial_state_prior")
            initial_state_prior = self._default_initial_state_prior(sdy)

        if isinstance(initial_state_prior, R.NormalPrior):
            dim = 2 * self._nseasons
            mu = np.zeros(dim)
            sigma = initial_state_prior.sd
            Sigma = np.diag(np.ones(dim) * sigma ** 2)
            self._initial_state_prior = R.MvnPrior(mu, Sigma)
        else:
            self._initial_state_prior = initial_state_prior

        assert isinstance(self._initial_state_prior, R.MvnPrior)

        if level_precision_priors is None:
            if sdy is None:
                sdy = self._compute_sdy(y, "level_precision_priors")
            self._level_precision_priors = [R.SdPrior(
                sdy / 100, .1, upper_limit=sdy) for i in range(self.nseasons)]
        else:
            self._level_precision_priors = level_precision_priors
        msg = "level_precision_priors must be a list of R.SdPrior objects"
        if not isinstance(self._level_precision_priors, list):
            raise Exception(msg)
        for x in self._level_precision_priors:
            if not isinstance(x, R.SdPrior):
                raise Exception(msg)

        if slope_precision_priors is None:
            if sdy is None:
                sdy = self._compute_sdy(y, "slope_precision_priors")
            self._slope_precision_priors = [R.SdPrior(
                sdy / 100, .1, upper_limit=sdy) for i in range(self.nseasons)]
        else:
            self._slope_precision_priors = slope_precision_priors
        msg = "slope_precision_priors must be a list of R.SdPrior objects"
        if not isinstance(self._slope_precision_priors, list):
            raise Exception(msg)
        for x in self._slope_precision_priors:
            if not isinstance(x, R.SdPrior):
                raise Exception(msg)

        self._build_model()
        self._state_contribution = None

    @property
    def label(self):
        return f"SeasonalTrend({self._nseasons})"

    @property
    def nseasons(self):
        """
        The number of seasons in a full cycle.
        """
        return self._nseasons

    @property
    def state_dimension(self):
        """
        The number of elements in the full state vector.
        """
        return self._nseasons * 2

    @property
    def state_contribution(self):
        """
        The posterior distribution of the contribution of this state component
        to the mean of y.
        """
        return self._state_contribution

    def allocate_space(self, niter, time_dimension):
        self._sigma_level_draws = np.empty((niter, self.nseasons))
        self._sigma_slope_draws = np.empty((niter, self.nseasons))
        self._state_contribution = np.zeros((niter, time_dimension))

    def record_state(self, iteration, state_matrix):
        self._sigma_level_draws[iteration, :] = (
            self._state_model.sigma_level.to_numpy())
        self._sigma_slope_draws[iteration, :] = (
            self._state_model.sigma_slope.to_numpy())
        time_index = state_matrix.shape[1]
        nseasons = self.nseasons
        row_index = np.array(list(range(self.nseasons)) * int(
            np.ceil(time_index / nseasons)))[:time_index] + self.state_index
        col_index = range(state_matrix.shape[1])
        self._state_contribution[iteration, :] = state_matrix[
            row_index, col_index]

    def restore_state(self, iteration):
        self._state_model.set_sigma_level(self._sigma_level_draws[iteration, :])
        self._state_model.set_sigma_slope(self._sigma_slope_draws[iteration, :])

    def plot_state_contribution(
            self, fig, gridspec, time, burn=None, ylim=None, **kwargs):
        return self.plot_state_contribution_default(
            fig=fig, gridspec=gridspec, time=time, burn=burn,
            ylim=ylim, **kwargs)

    def _build_model(self):
        self._state_model = boom.GeneralSeasonalLLT(self._nseasons)
        self._state_model.set_initial_state_mean(
            boom.Vector(self._initial_state_prior.mu))
        self._state_model.set_initial_state_variance(
            boom.SpdMatrix(self._initial_state_prior.variance))

        level_precision_priors = [
            x.boom() for x in self._level_precision_priors]
        slope_precision_priors = [
            x.boom() for x in self._slope_precision_priors]

        sampler = boom.GeneralSeasonalLLTIndependenceSampler(
            self._state_model,
            level_precision_priors,
            slope_precision_priors,
            boom.GlobalRng.rng)
        self._state_model.set_method(sampler)

    def _default_initial_state_prior(self, sdy):
        dim = 2 * self._nseasons
        mean = np.zeros(dim)
        variance = np.diag(np.ones(dim) * sdy)
        return R.MvnPrior(mean, variance)

    def _compute_sdy(self, y, which_prior: str):
        if y is None:
            raise Exception(
                f"One of y, sdy, or {which_prior} must be specified.")
        return np.nanstd(y, ddof=1)

    def __repr__(self):
        return f"A GeneralSeasonalLLT model with {self.nseasons} seasons."

    def __getstate__(self):
        payload = self.__dict__.copy()
        del payload["_state_model"]
        return payload

    def __setstate__(self, payload):
        self.__dict__ = payload
        self._build_model()
