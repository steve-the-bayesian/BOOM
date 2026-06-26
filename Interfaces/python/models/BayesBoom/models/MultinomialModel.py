import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from .BaseModels import MixtureComponent
from .DirichletModel import DirichletPrior
from .boom_utils import (
    to_boom_vector,
    unique_match,
)


# ---------------------------------------------------------------------------
# Multinomial models
# ---------------------------------------------------------------------------

class MultinomialModel(MixtureComponent):
    """Python wrapper for boom.MultinomialModel."""

    def __init__(self, probs, categories=None):
        """
        Args:
          probs: A discrete probability distribution (non-negative, sums to 1).
          categories: Optional labels for each category.
        """
        self._probs = probs
        self._categories = range(len(probs)) if categories is None else categories
        self._boom_model = None
        self._boom_sampler = None
        self._prior = None

    @property
    def probs(self):
        if self._boom_model:
            self._probs = self._boom_model.probs.to_numpy()
        return self._probs

    @property
    def dim(self):
        if self._boom_model:
            return self._boom_model.dim
        return len(self._probs)

    def allocate_space(self, niter):
        self._prob_draws = np.empty((niter, self.dim))

    def record_draw(self, iteration):
        if not self._boom_model:
            raise Exception("Object contains no boom model.")
        self._prob_draws[iteration, :] = self.probs

    def set_prior(self, prior):
        self._prior = prior

    def default_prior(self):
        return DirichletPrior(np.ones(self.dim))

    def boom(self):
        import BayesBoom.boom as boom
        if self._boom_model:
            return self._boom_model

        self._boom_model = boom.MultinomialModel(to_boom_vector(self._probs))

        if not self._prior:
            self.set_prior(self.default_prior())

        if isinstance(self._prior, DirichletPrior):
            self._boom_sampler = boom.MultinomialDirichletSampler(
                self._boom_model,
                self._prior.boom(),
                boom.GlobalRng.rng)
        else:
            raise Exception(
                f"MultinomialModel has a prior of class {type(self._prior)}.  "
                "I don't know how to create a sampler.")

        self._boom_model.set_method(self._boom_sampler)
        return self._boom_model

    def sim(self, sample_size=1):
        rng = np.random.default_rng()
        return rng.choice(a=self._categories, size=sample_size, p=self._probs)

    def create_boom_data_builder(self, data=None):
        from BayesBoom.R.boom_data_builders import (
            UnlabelledCategoricalDataBuilder,
            LabelledCategoricalDataBuilder,
        )
        if isinstance(self._categories, range):
            return UnlabelledCategoricalDataBuilder(self.dim)
        return LabelledCategoricalDataBuilder(self._categories)

    def plot_components(self,
                        components,
                        burn=0,
                        style="ts",
                        levels=None,
                        fig=None,
                        ax=None,
                        **kwargs):
        from BayesBoom.R.density import Density
        S = len(components)
        K = len(self._categories)
        nrows = int(np.floor(np.sqrt(K)))
        ncols = int(np.ceil(K / nrows))
        if burn < 0:
            burn = 0

        niter = self._prob_draws.shape[0]
        probs = np.empty((niter - burn, S, self.dim))
        if levels is None:
            levels = [str(s) for s in self._categories]
        for s in range(S):
            probs[:, s, :] = np.array(components[s]._prob_draws[burn:, :])

        style = unique_match(style, ["ts", "boxplot", "barplot", "density"])

        if style in ("ts", "boxplot", "density"):
            iteration = range(burn, niter)
            if fig is None and ax is None:
                fig, ax = plt.subplots(nrows, ncols, sharex=True,
                                       squeeze=False, sharey=True)
            elif ax is None:
                ax = fig.subplots(nrows, ncols, sharex=True,
                                  squeeze=False, sharey=True)
            counter = 0
            for i in range(nrows):
                for j in range(ncols):
                    if counter < K:
                        if style == "ts":
                            for s in range(S):
                                ax[i, j].plot(iteration, probs[:, s, counter],
                                              label=str(s))
                        elif style == "boxplot":
                            ax[i, j].boxplot(probs[:, :, counter])
                        elif style == "density":
                            for s in range(S):
                                den = Density(probs[:, s, counter])
                                den.plot(ax=ax[i, j], label=str(s))
                        ax[i, j].set_title(levels[counter])
                        if style != "boxplot":
                            ax[i, j].legend()
                        counter += 1

        elif style == "barplot":
            if fig is None and ax is None:
                fig, ax = plt.subplots(1, 1)
            elif ax is None:
                ax = fig.subplots(1, 1)
            mean_probs = probs.mean(axis=0)
            states = np.arange(S)
            cum_probs = np.zeros(S)
            grays = np.arange(K) / K
            for k in range(K):
                ax.bar(states, mean_probs[:, k], bottom=cum_probs,
                       width=.6, color=str(grays[k]))
                cum_probs += mean_probs[:, k]
            if S < 10:
                ax.set_xticks(np.arange(S), [str(s) for s in range(S)])
        else:
            raise Exception(f"Style argument '{style}' unrecognized.")


class MultilevelMultinomialModel(MixtureComponent):
    """Python wrapper for boom.MultilevelMultinomialModel."""

    def __init__(self, taxonomy, sep="/"):
        import BayesBoom.boom as boom
        self._sep = sep
        if isinstance(taxonomy, boom.Taxonomy):
            self._boom_taxonomy = taxonomy
        else:
            self._boom_taxonomy = boom.Taxonomy(taxonomy, sep)
        self._boom_model = None
        self._boom_sampler = None
        self._model_levels = None
        self._draws = None

    def probs(self, level=""):
        if self._boom_model is None:
            self.boom()
        if level:
            return self._boom_model.conditional_model(
                level, self._sep).probs.to_numpy()
        return self._boom_model.top_level_model.probs.to_numpy()

    def prob_draws(self, parent_level="", conditional=False, burn=0):
        """
        Return a DataFrame of posterior draws for the given taxonomy level.

        Args:
          parent_level: Parent node of the taxonomy.  Empty string means the
            top level.
          conditional: If True, return conditional probabilities that sum to 1
            within each draw.
          burn: Number of MCMC iterations to discard.
        """
        levels = self._boom_taxonomy.child_levels(parent_level)
        draws = pd.DataFrame(self._draws[parent_level], columns=levels)

        if conditional:
            return draws

        while parent_level:
            parent_level, child = self._boom_taxonomy.pop_level(parent_level)
            parent_probs = self._draws[parent_level]
            parent_prob_levels = self._boom_taxonomy.child_levels(parent_level)
            parent_probs = parent_probs[:, parent_prob_levels.index(child)]
            draws *= parent_probs.reshape(-1, 1)

        if burn > 0:
            draws = draws[burn:, :]

        return draws

    def boom(self):
        import BayesBoom.boom as boom
        if self._boom_model is None:
            self._boom_model = boom.MultilevelMultinomialModel(
                self._boom_taxonomy)
            self._ensure_posterior_sampler()
        self._model_levels = self._boom_model.model_levels
        return self._boom_model

    def allocate_space(self, niter):
        self.boom()
        params = self._boom_model.parameters
        self._draws = {
            x[0]: np.empty((niter, x[1].size))
            for x in zip(self._model_levels, params)
        }

    def record_draw(self, iteration):
        for draw in zip(self._model_levels, self._boom_model.parameters):
            self._draws[draw[0]][iteration, :] = draw[1].to_numpy()

    def create_boom_data_builder(self, data=None):
        from BayesBoom.R.boom_data_builders import MultilevelCategoricalDataBuilder
        return MultilevelCategoricalDataBuilder(self._boom_taxonomy, self._sep)

    def plot_components(self,
                        components,
                        burn=0,
                        style="ts",
                        fig=None,
                        ax=None,
                        parent_level="",
                        conditional=False,
                        **kwargs):
        from BayesBoom.R.density import Density
        from BayesBoom.R.plots import ensure_ax
        S = len(components)
        if burn < 0:
            burn = 0

        niter = self._draws[self._model_levels[0]].shape[0]
        levels = self._boom_taxonomy.child_levels(parent_level)
        K = len(levels)
        draws = np.empty((niter - burn, S, K))
        for s in range(S):
            draws[:, s, :] = components[s].prob_draws(
                burn=burn, parent_level=parent_level,
                conditional=conditional)

        style = unique_match(style, ["ts", "boxplot", "barplot", "density"])

        if style in ("ts", "boxplot", "density"):
            nrows = int(np.floor(np.sqrt(K)))
            ncols = int(np.ceil(K / nrows))
            iteration = range(burn, niter)
            if fig is None and ax is None:
                fig, ax = plt.subplots(nrows, ncols, sharex=True,
                                       squeeze=False, sharey=True)
            elif ax is None:
                ax = fig.subplots(nrows, ncols, sharex=True,
                                  squeeze=False, sharey=True)
            counter = 0
            for i in range(nrows):
                for j in range(ncols):
                    if counter < K:
                        if style == "ts":
                            for s in range(S):
                                ax[i, j].plot(iteration, draws[:, s, counter],
                                              label=str(s))
                        elif style == "boxplot":
                            ax[i, j].boxplot(draws[:, :, counter])
                        elif style == "density":
                            for s in range(S):
                                den = Density(draws[:, s, counter])
                                den.plot(ax=ax[i, j], label=str(s))
                        ax[i, j].set_title(levels[counter])
                        if style != "boxplot":
                            ax[i, j].legend()
                        counter += 1

        elif style == "barplot":
            fig, ax = ensure_ax(fig, ax)
            mean_probs = draws.mean(axis=0)
            states = np.arange(S)
            cum_probs = np.zeros(S)
            grays = np.arange(K) / K
            for k in range(K):
                ax.bar(states, mean_probs[:, k], bottom=cum_probs,
                       width=.6, color=str(grays[k]))
                cum_probs += mean_probs[:, k]
            if S < 10:
                ax.set_xticks(np.arange(S), [str(s) for s in range(S)])
        else:
            raise Exception(f"Style '{style}' unrecognized.")

        return fig, ax

    def _ensure_posterior_sampler(self):
        import BayesBoom.boom as boom
        self._boom_sampler = boom.MultilevelMultinomialPosteriorSampler(
            self._boom_model, 1.0)
        self._boom_model.set_method(self._boom_sampler)
