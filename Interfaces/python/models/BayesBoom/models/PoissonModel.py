import numpy as np
import matplotlib.pyplot as plt

from .BaseModels import MixtureComponent
from .boom_utils import unique_match
from .GammaModel import GammaModel

# ---------------------------------------------------------------------------
# Poisson model
# ---------------------------------------------------------------------------

class PoissonModel(MixtureComponent):
    """Python wrapper for boom.PoissonModel."""

    def __init__(self, mean=1):
        self._lambda = mean
        self._prior = None
        self._lambda_draws = None
        self._boom_model = None
        self._boom_sampler = None

    @property
    def mean(self):
        if self._boom_model is not None:
            self._lambda = self._boom_model.mean
        return self._lambda

    def set_prior(self, prior):
        self._prior = prior

    def default_prior(self):
        return GammaModel(1, 1)

    def boom(self):
        if self._boom_model is not None:
            return self._boom_model

        import BayesBoom.boom as boom
        self._boom_model = boom.PoissonModel(self._lambda)
        if self._prior is None:
            self.set_prior(self.default_prior())

        if isinstance(self._prior, GammaModel):
            self._boom_sampler = boom.PoissonGammaSampler(
                self._boom_model,
                self._prior.boom(),
                boom.GlobalRng.rng)
        else:
            raise Exception(
                f"PoissonModel has a prior of class {type(self._prior)}.  "
                "I don't know how to create a sampler.")

        self._boom_model.set_method(self._boom_sampler)
        return self._boom_model

    def allocate_space(self, niter):
        self._lambda_draws = np.empty(niter)

    def record_draw(self, iteration):
        self._lambda_draws[iteration] = self._boom_model.mean

    def create_boom_data_builder(self, data=None):
        from BayesBoom.R.boom_data_builders import IntDataBuilder
        return IntDataBuilder()

    def plot_components(self,
                        components,
                        burn=0,
                        style="ts",
                        fig=None,
                        ax=None,
                        **kwargs):
        from BayesBoom.R.density import Density
        S = len(components)
        if burn < 0:
            burn = 0

        niter = self._lambda_draws.shape[0]
        draws = np.empty((niter - burn, S))
        for s in range(S):
            draws[:, s] = components[s]._lambda_draws[burn:]

        if fig is None and ax is None:
            fig, ax = plt.subplots(1, 1)
        elif ax is None:
            ax = fig.subplots(1, 1)

        style = unique_match(style, ["ts", "density", "boxplot"])

        if style == "ts":
            iteration = range(burn, niter)
            for s in range(S):
                ax.plot(iteration, draws[:, s], label=str(s))
            ax.legend(loc="upper right")
        elif style == "density":
            for s in range(S):
                den = Density(draws[:, s])
                den.plot(ax=ax, label=str(s))
            ax.legend(loc="upper right")
        elif style == "boxplot":
            ax.boxplot(draws)
        else:
            raise Exception(f"Style argument '{style}' unrecognized.")

        return fig, ax
