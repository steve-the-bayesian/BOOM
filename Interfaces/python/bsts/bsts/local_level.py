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
        if sigma_prior is None:
            if sdy is None:
                sdy = np.std(y)
            sigma_prior = R.SdPrior(sigma_guess=.01 * sdy,
                                    sample_size=.01,
                                    upper_limit=sdy)
            if not isinstance(sigma_prior, R.SdPrior):
                raise Exception("sigma_prior should be an R.SdPrior.")

        if initial_state_prior is None:
            if initial_y is None:
                initial_y = y[0]
            if sdy is None:
                sdy = np.std(y)
            initial_y = float(initial_y)
            sdy = float(sdy)
            initial_state_prior = boom.GaussianModel(initial_y, sdy**2)
        if not isinstance(initial_state_prior, boom.GaussianModel):
            raise Exception(
                "initial_state_prior should be a boom.GaussianModel.")

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
        self._state_contribution = None

    def __repr__(self):
        return f"Local level with sigma = {self._state_model.sigma}"

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
        self.state_contribution[iteration, :] = state_matrix[
            self._state_index, :]

    @property
    def state_contribution(self):
        return self._state_contribution
