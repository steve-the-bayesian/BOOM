import BayesBoom.boom as boom
import numpy as np
import BayesBoom.R as R
from .state_models import StateModel


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

        self._state_contribution = None

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

    @property
    def state_contribution(self):
        return self._state_contribution

    def allocate_space(self, niter, time_dimension):
        self.sigma_draws = np.zeros(niter)
        self._state_contribution = np.zeros((niter, time_dimension))

    def record_state(self, iteration, state_matrix):
        self.sigma_draws[iteration] = self._state_model.sigma
        self._state_contribution[iteration, :] = state_matrix[
            self._state_index, :]

    def plot_state_contribution(
            self, fig, gridspec, time, burn, ylim, **kwargs):
        if self.nseasons == 7 and self.season_duration == 1:
            return self._plot_day_of_week_cycle(
                fig=fig, gridspec=gridspec, time=time, burn=burn,
                ylim=ylim, **kwargs)
        else:
            return self.plot_state_contribution_default(
                fig=fig, gridspec=gridspec, time=time, burn=burn,
                ylim=ylim, **kwargs)

    @staticmethod
    def _default_sigma_prior(sdy):
        """
        The default prior to use for the innovation standard deviation.
        """
        return R.SdPrior(.01 * sdy, upper_limit=sdy)

    def _default_initial_state_prior(self, sdy):
        """
        The default prior to use for the initial state vector.
        """
        dim = self.nseasons - 1
        return boom.MvnModel(
            boom.Vector(np.zeros(dim).astype(float)),
            boom.SpdMatrix(np.diag(np.full(dim, float(sdy)))))

    def _plot_day_of_week_cycle(
            self, fig, gridspec, burn, time, ylim, **kwargs):
        spec = gridspec.subgridspec(3, 3)
        season = 0
        for i in range(3):
            for j in range(3):
                if (j == 2) and (i < 2):
                    pass
                else:
                    ax = fig.add_subplot(spec)
                    time_subset = time[season::7]
                    R.PlotDynamicDistribution(
                        curves=self.state_contribution[burn:, season::7],
                        timestamps=time_subset,
                        ax=ax,
                        ylim=ylim,
                        **kwargs)
                    ax.tick_params(
                        bottom=False,
                        top=False,
                        left=False,
                        right=False,
                        labelbottom=False,
                        labeltop=False,
                        labelleft=False,
                        labelright=False,
                    )
                    if (i == 2):
                        ax.tick_params(
                            bottom=True,
                            labelbottom=True)
                    elif (i == 0) and (j == 2):
                        ax.tick_params(
                            top=True,
                            labeltop=True)

                    if j == 0:
                        ax.tick_params(
                            left=True,
                            labelleft=True)
        return None
