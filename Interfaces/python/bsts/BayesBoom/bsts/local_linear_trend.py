import BayesBoom.boom as boom
import numpy as np
from .state_models import StateModel
import BayesBoom.R as R


class LocalLinearTrendStateModel(StateModel):
    """
    The local linear trend model extends the local level model by including a
    slope parameter 'delta'.  The model is

             y[t] = mu[t] + error[t]
          mu[t+1] = mu[t] + delta[t] + innovation0[t]
       delta[t+1] = delta[t] + innovation1[t]

    The level component is "mu".  At each time point the mean changes by an
    expected amount delta (which is also time varying).  In this sense delta is
    the "local slope".

    The parameters of this model are the standard deviations of the innovation
    terms (which are modeled as independent zero-mean Gaussians).
    """

    def __init__(self,
                 y=None,
                 level_sigma_prior: R.SdPrior = None,
                 slope_sigma_prior: R.SdPrior = None,
                 initial_level_prior: R.NormalPrior = None,
                 initial_slope_prior: R.NormalPrior = None,
                 sdy: float = None,
                 initial_y: float = None):
        """
        Args:
          y:  The time series to be modeled.
          level_sigma_prior: Prior distribution on the standard deviation of
            the "level" innovations.
          slope_sigma_prior: Prior distribution on the standard deviation of
            the "slope" innovations.
          initial_level_prior: Prior distribution on the value of the level at
            time 0.
          initial_slope_prior: Prior distribution on the value of the slope at
            time 0.
          sdy:  The standard deviation of the time series being modeled.
          initial_y: The value of y at time 0.

        Details:
          There are three ways to build this object.
          (1) Pass all four prior objects.
          (2) Pass zero or more prior objects, sdy, and initial_y.
          (3) Pass y.

          Option 3 is the most convenient.  Option 1 offers the most control.
        """
        self._validate_priors(level_sigma_prior, slope_sigma_prior, y, sdy)
        self._validate_initial_distributions(
            initial_level_prior, initial_slope_prior, y, sdy, initial_y)
        self._state_contribution = None
        self._build_state_model()

    def __repr__(self):
        return (
            "A LocalLinearTrendStateModel with level_sd = "
            f"{self._state_model.sigma_level} and slope_sd = "
            f"{self._state_model.sigma_slope}."
        )

    @property
    def label(self):
        return "Trend (local linear)"

    @property
    def state_dimension(self):
        return 2

    @property
    def state_error_dimension(self):
        return 2

    @property
    def state_contribution(self):
        return self._state_contribution

    def allocate_space(self, niter, time_dimension):
        self.sigma_level = np.empty(niter)
        self.sigma_slope = np.empty(niter)
        self._state_contribution = np.empty((niter, time_dimension))

    def record_state(self, iteration, state_matrix):
        self.sigma_level[iteration] = self._state_model.sigma_level
        self.sigma_slope[iteration] = self._state_model.sigma_slope
        self._state_contribution[iteration, :] = state_matrix[
            self._state_index, :]

    def restore_state(self, iteration):
        self._state_model.set_sigma_level(self.sigma_level[iteration])
        self._state_model.set_sigma_slope(self.sigma_slope[iteration])

    def _validate_priors(self, level_sigma_prior, slope_sigma_prior, y, sdy):
        if level_sigma_prior is None:
            sdy = self._compute_sdy(sdy, y, "level_sigma_prior")
            level_sigma_prior = R.SdPrior(
                sigma_guess=.01 * sdy,
                upper_limit=sdy)
        if not isinstance(level_sigma_prior, R.SdPrior):
            raise Exception("Unexpected type for level_sigma_prior.")

        if slope_sigma_prior is None:
            sdy = self._compute_sdy(sdy, y, "slope_sigma_prior")
            slope_sigma_prior = R.SdPrior(
                sigma_guess=.01 * sdy,
                upper_limit=sdy)
        if not isinstance(slope_sigma_prior, R.SdPrior):
            raise Exception("Unexpected type for slope_sigma_prior.")

        self._level_sigma_prior = level_sigma_prior
        self._slope_sigma_prior = slope_sigma_prior

    def _validate_initial_distributions(
            self, initial_level_prior, initial_slope_prior,
            y, sdy, initial_y):
        if initial_level_prior is None:
            sdy = self._compute_sdy(sdy, y, "initial_level_prior")
            if initial_y is None:
                if y is None:
                    raise Exception(
                        "One of initial_y, y, or initial_level_prior must be "
                        "specified.")
                else:
                    initial_y = y[0]
            initial_level_prior = R.NormalPrior(initial_y, sdy)
        if not isinstance(initial_level_prior, R.NormalPrior):
            raise Exception("Unexpected type for initial_level_prior.")
        self._initial_level_prior = initial_level_prior

        if initial_slope_prior is None:
            sdy = self._compute_sdy(sdy, y, "initial_slope_prior")
            initial_slope_prior = R.NormalPrior(0, sdy)
        if not isinstance(initial_slope_prior, R.NormalPrior):
            raise Exception("Unexpected type for initial_slope_prior.")
        self._initial_slope_prior = initial_slope_prior

    def _build_state_model(self):
        self._state_model = boom.LocalLinearTrendStateModel()
        self._set_posterior_sampler()
        self._set_initial_distribution()

    def _set_posterior_sampler(self):
        """
        A utility called by the constructor.  See the __init__ method for
        argument documentation.
        """

        self._state_model.set_posterior_sampler(
            self._level_sigma_prior.create_chisq_model(),
            self._level_sigma_prior.upper_limit,
            self._slope_sigma_prior.create_chisq_model(),
            self._slope_sigma_prior.upper_limit,
            boom.GlobalRng.rng)

    def _set_initial_distribution(self):
        """
        A utility called by the constructor.  See the __init__ method for
        argument documentation.
        """
        initial_state_mean = np.array([
            self._initial_level_prior.mean,
            self._initial_slope_prior.mean,
        ])
        self._state_model.set_initial_state_mean(
            boom.Vector(initial_state_mean))
        initial_sd = np.array([
            self._initial_level_prior.sd,
            self._initial_slope_prior.sd,
        ])
        initial_variance = np.diag(initial_sd * initial_sd)
        self._state_model.set_initial_state_variance(
            boom.SpdMatrix(initial_variance))

    @staticmethod
    def _compute_sdy(sdy, y, which_prior):
        """
        Return the standard deviation of y, computing it if and only if needed.
        """
        if sdy is None:
            if y is None:
                raise Exception(
                    f"One of y, sdy, or {which_prior} must be specified.")
            else:
                sdy = np.nanstd(y, ddof=1)
        return sdy

    def __getstate__(self):
        payload = self.__dict__.copy()
        del payload["_state_model"]
        return payload

    def __setstate__(self, payload):
        self.__dict__ = payload
        self._build_state_model()
