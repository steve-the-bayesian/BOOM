import numpy as np
import BayesBoom.boom as boom
import BayesBoom.R as R
from numbers import Number
from .state_models import StateModel


class ArStateModel(StateModel):
    """
    Latent state described by a stationary AR(p) process.
    """

    def __init__(self,
                 y,
                 lags: int = 1,
                 sigma_prior: R.SdPrior = None,
                 initial_state_prior: R.MvnPrior = None,
                 sdy: float = None):
        """
        Args:
          y:  The time series to be modeled.
          lags:  The number of lags in the autoregressive process.
          sigma_prior: The prior distribution for the standard deviation
            parameter in the AR1 process.
          initial_state_prior: Prior distribution for the value of the state at
            time 0.
          sdy: The standard deviation of the time series to be modeled.  This
            argument is ignored if y is provided.
        """
        self._lags = lags
        self._state_contribution = None
        if sdy is None:
            sdy = np.nanstd(y)

        self._validate_prior(sigma_prior, sdy)
        self._validate_initial_distribution(initial_state_prior, sdy)
        self._build_state_model()

    @property
    def label(self):
        return f"AR({self._lags})"

    @property
    def lags(self):
        return self._lags

    @property
    def state_dimension(self):
        return self._lags

    @property
    def state_error_dimension(self):
        return 1

    @property
    def state_contribution(self):
        return self._state_contribution

    def allocate_space(self, niter, time_dimension):
        self._sigma = np.empty(niter)
        self._ar_coefficients = np.empty((niter, self.state_dimension))
        self._state_contribution = np.empty((niter, time_dimension))

    def record_state(self, iteration, state_matrix):
        self._sigma[iteration] = self._state_model.sigma
        self._ar_coefficients[iteration, :] = self._state_model.phi.to_numpy()
        self._state_contribution[iteration, :] = state_matrix[
            self._state_index, :]

    def restore_state(self, iteration):
        self._state_model.set_sigma(self._sigma[iteration])
        self._state_model.set_phi(R.to_boom_vector(
            self._ar_coefficients[iteration, :]))

    def _validate_prior(self, prior, sdy):
        if prior is None:
            prior = R.SdPrior(sdy * .01, upper_limit=sdy)
        if not isinstance(prior, R.SdPrior):
            raise Exception("Wrong type for prior.")
        self._sigma_prior = prior

    def _validate_initial_distribution(self, initial_state_prior, sdy):
        if initial_state_prior is None:
            dim = self.state_dimension
            initial_state_prior = R.MvnPrior(np.zeros(dim), np.eye(dim) * sdy)
        if not isinstance(initial_state_prior, R.MvnPrior):
            raise Exception("Wrong type for initial_state_prior.")
        if initial_state_prior.dim != self.state_dimension:
            raise Exception(
                f"Initial_state_prior dimension was {initial_state_prior.dim}."
                f"  State dimension is {self.state_dimension}."
            )
        self._initial_state_prior = initial_state_prior

    def _build_state_model(self):
        self._state_model = boom.ArStateModel(self._lags)
        self._set_posterior_sampler()
        self._set_initial_distribution()

    def _set_posterior_sampler(self):
        sampler = boom.ArPosteriorSampler(
            self._state_model,
            self._sigma_prior.boom(),
            boom.GlobalRng.rng)
        limit = self._sigma_prior.upper_limit
        if np.isfinite(limit) and limit > 0:
            sampler.set_sigma_upper_limit(limit)
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


class SpikeSlabArPrior:
    def __init__(self,
                 lags,
                 sdy=None,
                 prior_inclusion_probabilities=None,
                 prior_mean=None,
                 prior_sd=None,
                 prior_df=1,
                 expected_r2=0.5,
                 sigma_upper_limit=np.Inf,
                 max_flips=np.Inf):
        self._lags = int(lags)
        if self._lags < 0:
            raise Exception("'lags' argument must be non-negative.")

        self._max_flips = max_flips

        if prior_inclusion_probabilities is None:
            prior_inclusion_probabilities = np.geomspace(
                .8, .8 ** self._lags, num=self._lags)
        self._prior_inclusion_probabilities = np.array(
            prior_inclusion_probabilities)

        if prior_mean is None:
            self._prior_mean = np.zeros(self._lags)
        else:
            self._prior_mean = np.array(prior_mean, dtype=float)
        if len(self._prior_mean) != self._lags:
            raise Exception("prior_mean argument must have length 'lags'.")

        if prior_sd is None:
            self._prior_sd = np.geomspace(
                .5, .5 * .8 ** (self._lags - 1), num=self._lags)
        elif isinstance(self._prior, Number):
            self._prior_sd = np.array([prior_sd] * self._lags)
        else:
            self._prior_sd = np.array(prior_sd, dtype=float)

        sigsq_guess = sdy * expected_r2
        self._residual_precision = R.SdPrior(
            sigma_guess=np.sqrt(sigsq_guess),
            sample_size=prior_df,
            upper_limit=sigma_upper_limit)

    def __repr__(self):
        f"""
        SpikeSlabArPrior with...
        inclusion_probabilities: {self._prior_inclusion_probabilities}
        max flips: {self._max_flips}
        prior mean: {self._prior_mean}
        prior sd: {self._prior_sd}
        residual precision prior: {self._residual_precision}
        """

    @property
    def spike(self):
        return boom.VariableSelectionPrior(
            R.to_boom_vector(
                self._prior_inclusion_probabilities))

    @property
    def slab(self):
        return boom.MvnModel(
            R.to_boom_vector(self._prior_mean),
            R.to_boom_spd(np.diag(self._prior_sd ** 2))
        )

    @property
    def residual_precision(self):
        return self._residual_precision

    @property
    def max_flips(self):
        return self._max_flips


class AutoArStateModel(ArStateModel):
    def __init__(self,
                 y,
                 lags: int = 1,
                 prior: SpikeSlabArPrior = None,
                 initial_state_prior: R.MvnPrior = None,
                 sdy: float = None,
                 **kwargs):
        if sdy is None:
            sdy = np.nanstd(y)
        super().__init__(y, lags, prior, initial_state_prior, sdy)

    @property
    def label(self):
        return f"AutoAr({self.lags})"

    def _validate_prior(self, prior, sdy):
        if prior is None:
            prior = SpikeSlabArPrior(
                lags=self.lags,
                sdy=sdy,
                sigma_upper_limit=sdy)

        if not isinstance(prior, SpikeSlabArPrior):
            raise Exception("Wrong type for 'prior'.")
        self._prior = prior

    def _set_posterior_sampler(self):
        sampler = boom.ArSpikeSlabSampler(
            self._state_model,
            self._prior.slab,
            self._prior.spike,
            self._prior.residual_precision.boom(),
            True,
            boom.GlobalRng.rng)
        limit = self._prior.residual_precision.upper_limit
        if limit > 0 and np.isfinite(limit):
            sampler.set_sigma_upper_limit(limit)
        max_flips = self._prior.max_flips
        if max_flips > 0 and np.isfinite(max_flips):
            sampler.limit_model_selection(max_flips)

        self._state_model.set_method(sampler)

    def __getstate__(self):
        payload = self.__dict__.copy()
        del payload["_state_model"]
        return payload

    def __setstate__(self, payload):
        self.__dict__ = payload
        self._build_state_model()
