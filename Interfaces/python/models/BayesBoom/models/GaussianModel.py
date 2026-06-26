import copy
import numpy as np
import matplotlib.pyplot as plt

from .BaseModels import DoubleModel, MixtureComponent
from .boom_utils import unique_match
from .GammaModel import GammaModel

# ---------------------------------------------------------------------------
# Normal model and related
# ---------------------------------------------------------------------------

class GaussianSuf:
    """Sufficient statistics for a scalar Gaussian model."""

    def __init__(self, data=None):
        self._sum = 0
        self._sumsq = 0
        self._n = 0
        if data is not None:
            self.update(data)

    def update(self, incremental_data):
        """Accumulate sufficient statistics from incremental_data."""
        y = np.array(incremental_data)
        self._sum += np.nansum(y)
        self._sumsq += np.nansum(y * y)
        self._n += np.sum(~np.isnan(incremental_data))

    def combine(self, other):
        """Add sufficient statistics from other into self (in-place)."""
        self._n += other._n
        self._sum += other._sum
        self._sumsq += other._sumsq

    def __iadd__(self, other):
        if isinstance(other, GaussianSuf):
            self.combine(other)
        else:
            self.update(other)
        return self

    def __add__(self, other):
        ans = copy.copy(self)
        ans += other
        return ans

    @property
    def sample_size(self):
        return self._n

    @property
    def sum(self):
        return self._sum

    @property
    def mean(self):
        return self._sum / self.sample_size if self.sample_size > 0 else 0.0

    @property
    def sumsq(self):
        return self._sumsq

    def centered_sumsq(self, center=None):
        if self.sample_size <= 0:
            return 0
        if center is None:
            center = self.mean
        n = self.sample_size
        return self.sumsq - 2 * center * self.sum + n * center ** 2

    @property
    def sample_sd(self):
        return np.sqrt(self.sample_variance)

    @property
    def sample_variance(self):
        n = self.sample_size
        if n < 2:
            return 0
        return self.centered_sumsq() / (n - 1)

    def __eq__(self, other):
        return self.__dict__ == other.__dict__


class NormalInverseGammaModel:
    """
    Normal-inverse-gamma conjugate prior for a Gaussian model.

    Encodes the joint prior  mu | sigma^2 ~ N(mean_guess, sigma^2 / kappa)
    and  sigma^2 ~ InverseGamma(df/2, ss/2)  where ss = df * sd_guess^2.
    """

    def __init__(self,
                 mean_guess,
                 mean_prior_sample_size,
                 sd_guess,
                 sd_prior_sample_size):
        self._mean = mean_guess
        self._mean_prior_sample_size = mean_prior_sample_size
        self._df = sd_prior_sample_size
        self._sd_estimate = sd_guess

    @property
    def sumsq(self):
        return self._sd_estimate ** 2 * self._df

    def gaussian_given_sigma(self, sigsq_parameter):
        """Return a boom.GaussianModelGivenSigma for the mean component."""
        import BayesBoom.boom as boom
        return boom.GaussianModelGivenSigma(
            sigsq_parameter,
            self._mean,
            self._mean_prior_sample_size)

    def chisq(self):
        """Return a boom.ChisqModel for the variance component."""
        import BayesBoom.boom as boom
        return boom.ChisqModel(self._df, self._sd_estimate)


class NormalModel(DoubleModel, MixtureComponent):
    """Scalar normal distribution, usable as a prior or mixture component."""

    def __init__(self,
                 mu: float = 0.0,
                 sigma: float = 1.0,
                 initial_value: float = None):
        self.mu = float(mu)
        self.sigma = float(sigma)
        self.initial_value = float(mu if initial_value is None else initial_value)
        self._boom_model = None
        self._prior = None
        self._mean_prior = None
        self._sd_prior = None

    @property
    def mean(self):
        return self.mu

    @property
    def sd(self):
        return self.sigma

    @property
    def variance(self):
        return self.sigma ** 2

    def set_prior(self, prior):
        if isinstance(prior, NormalInverseGammaModel):
            self._prior = prior
        elif isinstance(prior, NormalModel):
            self._mean_prior = prior
        elif isinstance(prior, GammaModel):
            self._sd_prior = prior

    def default_prior(self):
        return NormalInverseGammaModel(0, 1, 1, 1)

    def boom(self):
        import BayesBoom.boom as boom
        if self._boom_model is None:
            self._boom_model = boom.GaussianModel(self.mu, self.sigma)

            if not self._prior:
                self.set_prior(self.default_prior())

            if (self._prior is not None
                    and isinstance(self._prior, NormalInverseGammaModel)):
                boom_sampler = boom.GaussianConjugateSampler(
                    self._boom_model,
                    self._prior.gaussian_given_sigma(
                        self._boom_model.sigsq_parameter),
                    self._prior.chisq())
                self._boom_model.set_method(boom_sampler)
            else:
                raise Exception(
                    "Not sure how to build the posterior sampler for "
                    f"prior of type {type(self._prior)}.")

        return self._boom_model

    def allocate_space(self, niter):
        self._mu_draws = np.empty(niter)
        self._sigma_draws = np.empty(niter)

    def record_draw(self, iteration):
        self._mu_draws[iteration] = self._boom_model.mu
        self._sigma_draws[iteration] = self._boom_model.sigma

    def create_boom_data_builder(self, data=None):
        from BayesBoom.R.boom_data_builders import DoubleDataBuilder
        return DoubleDataBuilder()

    def plot_components(self,
                        components,
                        burn=0,
                        style="ts",
                        fig=None,
                        ax=None,
                        params="both",
                        **kwargs):
        params = unique_match(params, ["mean", "sd", "both"])
        style = unique_match(style, ["ts", "boxplot", "histogram", "density"])
        S = len(components)

        numplots = 2 if params == "both" else 1
        if fig is None and ax is None:
            fig, ax = plt.subplots(1, numplots)
        elif ax is None:
            ax = fig.subplots(1, numplots)

        if params == "both":
            self._plot_params(components=components, burn=burn, style=style,
                              fig=fig, ax=ax[0], what="means", **kwargs)
            self._plot_params(components=components, burn=burn, style=style,
                              fig=fig, ax=ax[1], what="sds", **kwargs)
        elif params == "mean":
            self._plot_params(components=components, burn=burn, style=style,
                              fig=fig, ax=ax, what="means", **kwargs)
        elif params == "sd":
            self._plot_params(components=components, burn=burn, style=style,
                              fig=fig, ax=ax, what="sds", **kwargs)
        else:
            raise Exception(f"Unknown value of params: {params}")

        return fig, ax

    def _plot_params(self, components, burn=0, style="ts", fig=None, ax=None,
                     what="means", **kwargs):
        from BayesBoom.R.plots import ensure_ax
        from BayesBoom.R.density import Density
        fig, ax = ensure_ax(fig, ax)
        style = unique_match(style, ["ts", "boxplot", "density"])
        what = unique_match(what, ["means", "sds"])
        S = len(components)
        niter = components[0]._mu_draws.shape[0]
        if burn < 0:
            burn = 0
        params = np.empty((niter - burn, S))
        if what == "means":
            for s in range(S):
                params[:, s] = components[s]._mu_draws[burn:]
            label = "means"
        else:
            for s in range(S):
                params[:, s] = components[s]._sigma_draws[burn:]
            label = "Standard Deviations"

        if style == "ts":
            iteration = range(burn, niter)
            for s in range(S):
                ax.plot(iteration, params[:, s], label=s)
            ax.legend()
            ax.set_xlabel("Iteration")
            ax.set_ylabel(label)
        elif style == "density":
            for s in range(S):
                den = Density(params[:, s])
                den.plot(ax=ax, label=s)
            ax.legend()
            ax.set_xlabel(label)
            ax.set_ylabel("density")
        elif style == "boxplot":
            ax.boxplot(params)
            ax.set_ylabel(label)
            ax.set_xlabel("mixture component")

        return fig, ax

    def __getstate__(self):
        return {
            "mu": self.mean,
            "sigma": self.sd,
            "initial_value": self.initial_value,
        }

    def __setstate__(self, payload):
        self.mu = payload["mu"]
        self.sigma = payload["sigma"]
        self.initial_value = payload["initial_value"]


GaussianModel = NormalModel
