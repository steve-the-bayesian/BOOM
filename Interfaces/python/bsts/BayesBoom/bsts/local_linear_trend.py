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
                 initial_level_prior: boom.GaussianModel = None,
                 initial_slope_prior: boom.GaussianModel = None,
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
        self._state_model = boom.LocalLinearTrendStateModel()
        self._set_posterior_sampler(y, level_sigma_prior, slope_sigma_prior,
                                    sdy)
        self._set_initial_distribution(
            y, initial_level_prior, initial_slope_prior,
            sdy, initial_y)
        self._state_contribution = None

    def __repr__(self):
        return (
            "A LocalLinearTrendStateModel with level_sd = "
            f"{self._state_model.sigma_level} and slope_sd = "
            f"{self._state_model.sigma_slope}."
            )

    @property
    def state_dimension(self):
        return 2

    @property
    def state_error_dimension(self):
        return 2

    def allocate_space(self, niter, time_dimension):
        self.sigma_level = np.empty(niter)
        self.sigma_slope = np.empty(niter)
        self._state_contribution = np.empty((niter, time_dimension))

    def record_state(self, iteration, state_matrix):
        self.sigma_level[iteration] = self._state_model.sigma_level
        self.sigma_slope[iteration] = self._state_model.sigma_slope
        self.state_contribution[iteration, :] = state_matrix[
            self._state_index, :]

    def _set_posterior_sampler(
            self, y, level_sigma_prior, slope_sigma_prior, sdy):
        """
        A utility called by the constructor.  See the __init__ method for
        argument documentation.
        """
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
                sigma_guess=0.1 * sdy,
                upper_limit=sdy)
        if not isinstance(slope_sigma_prior, R.SdPrior):
            raise Exception("Unexpected type for slope_sigma_prior.")

        self._state_model.set_posterior_sampler(
            level_sigma_prior.create_chisq_model(),
            level_sigma_prior.upper_limit,
            slope_sigma_prior.create_chisq_model(),
            slope_sigma_prior.upper_limit,
            boom.GlobalRng.rng)

    def _set_initial_distribution(self, y, initial_level_prior,
                                  initial_slope_prior, sdy, initial_y):
        """
        A utility called by the constructor.  See the __init__ method for
        argument documentation.
        """
        if initial_level_prior is None:
            sdy = self._compute_sdy(sdy, y, "initial_level_prior")
            if initial_y is None:
                if y is None:
                    raise Exception(
                        "One of initial_y, y, or initial_level_prior must be "
                        "specified.")
                else:
                    initial_y = y[0]
            initial_level_prior = boom.GaussianModel(initial_y, sdy)
        if not isinstance(initial_level_prior, boom.GaussianModel):
            raise Exception("Unexpected type for initial_level_prior.")

        if initial_slope_prior is None:
            sdy = self._compute_sdy(sdy, y, "initial_slope_prior")
            initial_slope_prior = boom.GaussianModel(0, sdy)
        if not isinstance(initial_slope_prior, boom.GaussianModel):
            raise Exception("Unexpected type for initial_slope_prior.")

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
                sdy = np.nanstd(y)
        return sdy
