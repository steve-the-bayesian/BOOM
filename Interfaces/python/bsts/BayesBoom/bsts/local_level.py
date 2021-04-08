import BayesBoom.boom as boom
import numpy as np
import BayesBoom.R as R
from .state_models import StateModel


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
        self._validate_sigma_prior(sigma_prior, y, sdy)
        self._validate_initial_distributions(initial_state_prior, y, sdy,
                                             initial_y)
        self._build_state_model()
        self._state_contribution = None

    def __repr__(self):
        return f"Local level with sigma = {self._state_model.sigma}"

    @property
    def label(self):
        return "Trend (local level)"

    @property
    def state_dimension(self):
        return 1

    def allocate_space(self, niter, time_dimension):
        self.sigma_draws = np.zeros(niter)
        self._state_contribution = np.zeros((niter, time_dimension))

    def record_state(self, iteration, state_matrix):
        self.sigma_draws[iteration] = self._state_model.sigma
        if self._state_index < 0:
            raise Exception("Each state model must be told where its state"
                            "component begins in the global state vector.  "
                            "Try calling set_state_index.")
        self._state_contribution[iteration, :] = state_matrix[
            self._state_index, :]

    def restore_state(self, iteration):
        self._state_model.set_sigma(self.sigma_draws[iteration])

    @property
    def state_contribution(self):
        return self._state_contribution

    def _validate_sigma_prior(self, sigma_prior, y, sdy):
        if sigma_prior is None:
            if sdy is None:
                sdy = np.nanstd(y, ddof=1)
            sigma_prior = R.SdPrior(sigma_guess=.01 * sdy,
                                    sample_size=.01,
                                    upper_limit=sdy)
            if not isinstance(sigma_prior, R.SdPrior):
                raise Exception("sigma_prior should be an R.SdPrior.")
        self._sigma_prior = sigma_prior

    def _validate_initial_distributions(self, initial_state_prior, y, sdy,
                                        initial_y):
        if initial_state_prior is None:
            if initial_y is None:
                initial_y = y[0]
            if sdy is None:
                sdy = np.nanstd(y, ddof=1)
            initial_state_prior = R.NormalPrior(float(initial_y), float(sdy))
        if not isinstance(initial_state_prior, R.NormalPrior):
            raise Exception(
                "initial_state_prior should be an R.NormalPrior.")
        self._initial_state_prior = initial_state_prior

    def _build_state_model(self):
        self._state_model = boom.LocalLevelStateModel()
        self._state_model.set_initial_state_mean(self._initial_state_prior.mean)
        self._state_model.set_initial_state_variance(
            self._initial_state_prior.variance)
        innovation_precision_prior = boom.ChisqModel(
            self._sigma_prior.sigma_guess,
            self._sigma_prior.sample_size)
        state_model_sampler = self._state_model.set_posterior_sampler(
            innovation_precision_prior)
        state_model_sampler.set_sigma_upper_limit(
            self._sigma_prior.upper_limit)

    def __getstate__(self):
        payload = self.__dict__.copy()
        del payload["_state_model"]
        return payload

    def __setstate__(self, payload):
        self.__dict__ = payload
        self._build_state_model()
