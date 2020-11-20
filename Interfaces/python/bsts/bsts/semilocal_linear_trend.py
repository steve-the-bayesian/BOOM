import BayesBoom.boom as boom
import numpy as np
from .state_models import StateModel
import BayesBoom.R as R


class SemilocalLinearTrendStateModel(StateModel):
    """
    """

    def __init__(self,
                 y=None,
                 level_sigma_prior: R.SdPrior = None,
                 slope_mean_prior: R.NormalPrior = None,
                 slope_ar1_prior: R.Ar1CoefficientPrior = None,
                 slope_sigma_prior: R.SdPrior = None,
                 initial_level_prior: R.NormalPrior = None,
                 initial_slope_prior: R.NormalPrior = None,
                 sdy: float = None,
                 initial_y: float = None):
        """
        Args:
          y:  A numeric vector.  The time series to be modeled.
          level_sigma_prior: The prior distribution for the standard deviation
            of the increments in the level component of state.
          slope.mean_prior: The prior distribution for the mean of the AR1
            process for the slope component of state.
          slope_ar1_prior: The prior distribution for the ar1 coefficient in
            the slope component of state.
          slope_sigma_prior: The prior distribution for the standard deviation
            of the increments in the slope component of state.
          initial_level_prior: The prior distribution for the level component
            of state at the time of the first observation.
          initial_slope_prior: The prior distribution for the slope component
            of state at the time of the first observation.
          sdy: The standard deviation of y.  This can be ignored if y is
            provided, or if all the required prior distributions are supplied
            directly.
          initial.y: The initial value of y.  This can be omitted if y is
            provided.
        """

        # ----------------------------------------------------------------------
        # Validate all priors, generating default values as needed.
        if sdy is None:
            sdy = np.nanstd(y)
        if initial_y is None:
            initial_y = y[0]
        level_sigma_prior = self._validate_level_sigma_prior(
            level_sigma_prior, sdy)
        slope_mean_prior = self._validate_slope_mean_prior(
            slope_mean_prior, sdy)
        slope_ar1_prior = self._validate_slope_ar1_prior(
            slope_ar1_prior, sdy)
        slope_sigma_prior = self._validate_slope_sigma_prior(
            slope_sigma_prior, sdy)
        initial_level_prior = self._validate_initial_level_prior(
            initial_level_prior, initial_y, sdy)
        initial_slope_prior = self._validate_initial_slope_prior(
            initial_slope_prior, sdy)

        # ----------------------------------------------------------------------
        # Create the component models and samplers.
        level_model = boom.ZeroMeanGaussianModel(
            level_sigma_prior.initial_value)
        level_sampler = boom.ZeroMeanGaussianConjSampler(
            level_model,
            level_sigma_prior.boom(),
            boom.GlobalRng.rng)
        level_sampler.set_sigma_upper_limit(level_sigma_prior.upper_limit)
        level_model.set_method(level_sampler)

        slope_model = boom.NonzeroMeanAr1Model(
            slope_mean_prior.initial_value,
            slope_ar1_prior.initial_value,
            slope_sigma_prior.initial_value)

        slope_sampler = boom.NonzeroMeanAr1Sampler(
            slope_model,
            slope_mean_prior.boom(),
            slope_ar1_prior.boom(),
            slope_sigma_prior.boom())

        slope_sampler.set_sigma_upper_limit(slope_sigma_prior.upper_limit)
        if slope_ar1_prior.force_stationary:
            slope_sampler.force_stationary()
        if slope_ar1_prior.force_positive:
            slope_sampler.force_ar1_positive()

        # ----------------------------------------------------------------------
        # Create the state model.
        self._state_model = boom.SemilocalLinearTrendStateModel(
            level_model, slope_model)
        self._state_model.set_method(level_sampler)
        self._state_model.set_method(slope_sampler)

        # ----------------------------------------------------------------------
        # Set the initial distribution priors.
        self._state_model.set_initial_level_mean(
            initial_level_prior.mu)
        self._state_model.set_initial_level_sd(
            initial_level_prior.sigma)
        self._state_model.set_initial_slope_mean(
            initial_slope_prior.mu)
        self._state_model.set_initial_slope_sd(
            initial_slope_prior.sigma)

        self._state_contribution = None

    def __repr__(self):
        return (
            "A SemilocalLinearTrendStateModel with \n"
            f"level_sd = {self._state_model.level_sd} \n"
            f"slope_sd = {self._state_model.slope_sd}\n"
            f"slope_ar_coefficient = {self._state_model.slope_ar_coefficient}\n"
            f"slope_mean = {self._state_model.slope_mean}\n"
            )

    @property
    def state_dimension(self):
        return self._state_model.state_dimension

    @property
    def state_error_dimension(self):
        return self._state_model.state_error_dimension

    # TODO: change the names once they're defined in the pb11 module.
    def allocate_space(self, niter, time_dimension):
        self.level_sigma = np.empty(niter)
        self.slope_sigma = np.empty(niter)
        self.slope_ar1 = np.empty(niter)
        self.slope_mean = np.empty(niter)
        self._state_contribution = np.empty((niter, time_dimension))

    def record_state(self, iteration, state_matrix):
        self.level_sigma[iteration] = self._state_model.level_sd
        self.slope_ar1[iteration] = self._state_model.slope_ar_coefficient
        self.slope_mean[iteration] = self._state_model.slope_mean
        self._state_contribution[iteration, :] = state_matrix[
            self._state_index, :]

    @property
    def state_contribution(self):
        return self._state_contribution

    @staticmethod
    def _validate_level_sigma_prior(level_sigma_prior, sdy):
        if level_sigma_prior is None:
            level_sigma_prior = R.SdPrior(.01 * sdy, upper_limit=sdy)
        if not isinstance(level_sigma_prior, R.SdPrior):
            raise Exception("Wrong type passed for level_sigma_prior.  "
                            "Expected an R.SdPrior")
        return level_sigma_prior

    @staticmethod
    def _validate_slope_sigma_prior(slope_sigma_prior, sdy):
        if slope_sigma_prior is None:
            slope_sigma_prior = R.SdPrior(.01 * sdy, upper_limit=sdy)
        if not isinstance(slope_sigma_prior, R.SdPrior):
            raise Exception("Wrong type passed for slope_sigma_prior.  "
                            "Expected an R.SdPrior")
        return slope_sigma_prior

    @staticmethod
    def _validate_slope_ar1_prior(slope_ar1_prior, sdy):
        if slope_ar1_prior is None:
            slope_ar1_prior = R.Ar1CoefficientPrior()
        if not isinstance(slope_ar1_prior, R.Ar1CoefficientPrior):
            raise Exception("Wrong type passed for slope_ar1_prior.  "
                            "Expected an R.Ar1CoefficientPrior")
        return slope_ar1_prior

    @staticmethod
    def _validate_slope_mean_prior(slope_mean_prior, sdy):
        if slope_mean_prior is None:
            slope_mean_prior = R.NormalPrior(0, sdy)
        if not isinstance(slope_mean_prior, R.NormalPrior):
            raise Exception("Wrong type passed for slope_mean_prior.  "
                            "Expected an R.NormalPrior")
        return slope_mean_prior

    @staticmethod
    def _validate_initial_slope_prior(initial_slope_prior, sdy):
        if initial_slope_prior is None:
            initial_slope_prior = R.NormalPrior(0, sdy)
        if not isinstance(initial_slope_prior, R.NormalPrior):
            raise Exception("Wrong type for initial_slope_prior.  "
                            "Expected an R.NormalPrior.")
        return initial_slope_prior

    @staticmethod
    def _validate_initial_level_prior(initial_level_prior, initial_y, sdy):
        if initial_level_prior is None:
            initial_level_prior = R.NormalPrior(initial_y, sdy)
        if not isinstance(initial_level_prior, R.NormalPrior):
            raise Exception("Wrong type for initial_level_prior.  "
                            "Expected an R.NormalPrior.")
        return initial_level_prior
