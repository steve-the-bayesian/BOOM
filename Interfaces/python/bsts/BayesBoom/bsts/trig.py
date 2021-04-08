import numpy as np
import BayesBoom.boom as boom
import BayesBoom.R as R
from .state_models import StateModel


class TrigStateModel(StateModel):
    """
    Describe seasonal effects with harmonic functions rather than dummy
    variables.
    """

    def __init__(self, y, period, frequencies, sigma_prior=None,
                 initial_state_prior=None, sdy=None):
        """
        Args:
          y:  The time series to be modeled.
          period:  The number of time steps in a full cycle.
          frequencies: A vector of positive real numbers, giving the number of
            times a cycle repeats in a period.  One sine and one cosine term
            will be added to the state for each frequency.
          sigma_prior: The prior distribution for the standard deviations of
            the changes in the sinusoid coefficients at each new time point.
            This can be None (in which case a default prior will be used), or a
            single object of class R.SdPrior (which will be repeated for each
            sinusoid independently).
          initial_state_prior: The prior distribution for the values of the
            sinusoid coefficients at time 0.  This can either be None (in which
            case a default prior will be used), or an object of class
            R.MvnPrior.  If the prior is specified directly its dimension must
            be twice the number of frequencies.
          sdy: The standard deviation of the time series to be modeled.  This
            argument is ignored if y is provided.
        """
        self._period = period
        self._frequencies = frequencies
        if sdy is None:
            sdy = np.nanstd(y)

        self._validate_prior(sigma_prior, sdy)
        self._validate_initial_distribution(initial_state_prior, sdy)
        self._state_contribution = None
        self._build_state_model()

    @property
    def label(self):
        ans = "Trig["
        ans += f"{self._period:.2f}" + " | "
        for freq in self._frequencies:
            ans += f"{freq:.1f}, "
        ans = ans[:-2] + "]"
        return ans

    @property
    def state_dimension(self):
        return 2 * len(self._frequencies)

    @property
    def state_error_dimension(self):
        return self.state_dimension

    @property
    def state_contribution(self):
        return self._state_contribution

    def allocate_space(self, niter, time_dimension):
        self._sigma = np.empty(niter)
        self._state_contribution = np.empty((niter, time_dimension))

    def record_state(self, iteration, state_matrix):
        self._sigma[iteration] = self._state_model.error_distribution.sd
        begin = self._state_index
        end = begin + self.state_dimension
        state = state_matrix[begin:end, :]

        self._state_contribution[iteration, :] = (
            self._state_model.compute_state_contribution(state)
        ).to_numpy()

    def restore_state(self, iteration):
        self._state_model.error_distribution.set_sigma(
            self._sigma[iteration])

    def _validate_prior(self, sigma_prior, sdy):
        if sigma_prior is None:
            sigma_prior = R.SdPrior(sdy * .01, upper_limit=sdy)
        if not isinstance(sigma_prior, R.SdPrior):
            raise Exception("Wrong type for sigma_prior.")
        self._sigma_prior = sigma_prior

    def _validate_initial_distribution(self, initial_state_prior, sdy):
        if initial_state_prior is None:
            dim = self.state_dimension
            initial_state_prior = R.MvnPrior(
                np.zeros(dim),
                np.diag(np.ones(dim) * sdy ** 2))
        if not isinstance(initial_state_prior, R.MvnPrior):
            raise Exception("Wrong type for initial_state_prior.")
        if len(initial_state_prior.mean) != self.state_dimension:
            raise Exception(
                f"Initial_state_prior dimension was {initial_state_prior.dim}."
                f"  State dimension is {self.state_dimension}."
            )
        self._initial_state_prior = initial_state_prior

    def _build_state_model(self):
        self._state_model = boom.TrigStateModel(
            self._period, R.to_boom_vector(self._frequencies))
        self._set_posterior_sampler()
        self._set_initial_distribution()

    def _set_posterior_sampler(self):
        sampler = boom.ZeroMeanGaussianConjSampler(
            self._state_model.error_distribution,
            self._sigma_prior.boom(),
            boom.GlobalRng.rng)
        if (
                self._sigma_prior.upper_limit > 0
                and np.isfinite(self._sigma_prior.upper_limit)
        ):
            sampler.set_sigma_upper_limit(self._sigma_prior.upper_limit)
        self._state_model.set_method(sampler)

    def _set_initial_distribution(self):
        self._state_model.set_initial_state_mean(
            R.to_boom_vector(self._initial_state_prior.mean))
        self._state_model.set_initial_state_variance(
            R.to_boom_spd(self._initial_state_prior.variance))

    def __getstate__(self):
        payload = self.__dict__.copy()
        del payload["_state_model"]
        return payload

    def __setstate__(self, payload):
        self.__dict__ = payload
        self._build_state_model()
