import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .BaseModels import MixtureComponent
from .boom_utils import (
    to_boom_vector,
    to_boom_matrix,
    unique_match,
)

# ---------------------------------------------------------------------------
# Markov model and related
# ---------------------------------------------------------------------------

class MarkovConjugatePrior:
    """Conjugate prior for a MarkovModel."""

    def __init__(self, prior_transition_counts, prior_initial_counts=None):
        self._prior_transition_counts = prior_transition_counts
        self._prior_initial_counts = prior_initial_counts

    @property
    def prior_transition_counts(self):
        return self._prior_transition_counts

    @property
    def prior_initial_counts(self):
        return self._prior_initial_counts


class MarkovSuf:
    """Sufficient statistics for a Markov model."""

    def __init__(self, categorical_data_sequence=None, levels=None,
                 sort_levels=False):
        self._last_value = None
        self._initial_value = None
        self._levels = self._create_levels(
            levels, categorical_data_sequence, sort_levels)
        self._transition_counts = np.zeros(
            (self.num_levels, self.num_levels))
        if categorical_data_sequence is not None:
            self.increment(categorical_data_sequence)

    @property
    def num_levels(self):
        return len(self._levels)

    def increment(self, categorical_data_sequence):
        if self._initial_value is None:
            self._initial_value = np.array(categorical_data_sequence)[0]
            self._last_value = self._initial_value
            categorical_data_sequence = categorical_data_sequence[1:]

        categorical_data_sequence = pd.Categorical(
            categorical_data_sequence,
            categories=self._levels.index)
        numeric_codes = categorical_data_sequence.codes
        previous_codes = numeric_codes[:-1]
        next_codes = numeric_codes[1:]
        sample_size = previous_codes.shape[0]

        X0 = np.zeros((sample_size, self.num_levels))
        X0[np.arange(sample_size), previous_codes] = 1

        X1 = np.zeros((sample_size, self.num_levels))
        X1[np.arange(sample_size), next_codes] = 1

        self._transition_counts += X0.T @ X1
        last_value_code = self._levels[self._last_value]
        self._transition_counts[last_value_code, previous_codes[0]] += 1
        self._last_value = categorical_data_sequence[-1]

    @property
    def initial_value_counts(self):
        if not hasattr(self, "_initial_value_counts"):
            self._initial_value_counts = np.zeros(self.num_levels)
            if self._initial_value is not None:
                code = self._levels[self._initial_value]
                self._initial_value_counts[code] = 1.0
        return self._initial_value_counts

    @property
    def transition_counts(self):
        return pd.DataFrame(self._transition_counts,
                            index=self._levels.index,
                            columns=self._levels.index)

    def boom(self):
        import BayesBoom.boom as boom
        return boom.MarkovSuf(
            to_boom_matrix(self._transition_counts),
            to_boom_vector(self.initial_value_counts))

    def top_transitions(self, n, omit_self_transitions=False, probs=False):
        trans = self.transition_counts.copy()
        if probs:
            row_totals = trans.sum(axis=1)
            row_totals[row_totals <= 0] = 1
            trans = trans.div(row_totals, axis="rows")
        if omit_self_transitions:
            np.fill_diagonal(trans.values, 0)
        return trans.stack().nlargest(n)

    def _create_levels(self, levels, categorical_data_sequence, sort_levels):
        if (levels is None) and (categorical_data_sequence is None):
            raise Exception(
                "At least one of 'categorical_data_sequence' or 'levels' "
                "is required.")
        if levels is not None:
            level_info = np.unique(levels)
        else:
            level_info = np.unique(categorical_data_sequence)
        if sort_levels:
            level_info = np.sort(level_info)
        level_info = pd.Series(level_info, index=None)
        return pd.Series(level_info.index, index=level_info)

    def __add__(self, other):
        if np.any(self._levels != other._levels):
            raise Exception(
                "Cannot add sufficient statistics built with different levels.")
        ans = MarkovSuf(levels=self._levels, sort_levels=False)
        ans._transition_counts = self._transition_counts + other._transition_counts
        ans._initial_value_counts = (self.initial_value_counts
                                     + other.initial_value_counts)
        ans._last_value = other._last_value
        return ans

    def __iadd__(self, other):
        if np.any(self._levels != other._levels):
            raise Exception(
                "Cannot add sufficient statistics built with different levels.")
        self._transition_counts += other._transition_counts
        self._initial_value_counts = (self.initial_value_counts
                                      + other.initial_value_counts)
        self._last_value = other._last_value
        return self


class MarkovModel(MixtureComponent):
    """Python wrapper for boom.MarkovModel."""

    def __init__(self,
                 transition_matrix=None,
                 initial_distribution=None,
                 state_size=None,
                 categories=None):
        """
        Args:
          transition_matrix: Row-stochastic transition matrix.  If None, a
            uniform matrix is used and state_size must be given.
          initial_distribution: Distribution over initial state.  Defaults to
            uniform.
          state_size: Number of states; only used when transition_matrix is None.
          categories: Optional string labels for states.
        """
        if categories is not None:
            state_size = len(categories)

        if transition_matrix is None:
            if state_size is None:
                raise Exception(
                    "state_size must be given when transition_matrix is None.")
            transition_matrix = np.ones((state_size, state_size)) / state_size

        state_size = transition_matrix.shape[0]
        if (categories is not None) and (state_size != len(categories)):
            raise Exception(
                f"len(categories) = {len(categories)} does not match "
                f"state_size = {state_size}.")

        if initial_distribution is None:
            initial_distribution = np.ones(state_size) / state_size

        self._transition_matrix = transition_matrix
        self._initial_distribution = initial_distribution
        self._boom_model = None
        self._prior = None
        self._data = None
        self._categories = categories

    @property
    def state_size(self):
        return self._transition_matrix.shape[0]

    @property
    def transition_probabilities(self):
        if self._boom_model:
            self._transition_matrix = (
                self._boom_model.transition_probabilities.to_numpy())
        return self._transition_matrix

    @property
    def initial_distribution(self):
        if self._boom_model:
            self._initial_distribution = (
                self._boom_model.initial_distribution.to_numpy())
        return self._initial_distribution

    def set_prior(self, prior):
        if not isinstance(prior, MarkovConjugatePrior):
            raise Exception("Prior must be a MarkovConjugatePrior.")
        self._prior = prior

    def default_prior(self):
        S = self.state_size
        return MarkovConjugatePrior(np.ones((S, S)), np.ones(S))

    def boom(self):
        if self._boom_model is not None:
            return self._boom_model

        import BayesBoom.boom as boom
        self._boom_model = boom.MarkovModel(
            to_boom_matrix(self._transition_matrix),
            to_boom_vector(self._initial_distribution))

        if self._prior is None:
            self.set_prior(self.default_prior())

        if isinstance(self._prior, MarkovConjugatePrior):
            if self._prior.prior_initial_counts is None:
                self._boom_sampler = boom.MarkovConjugateSampler(
                    self._boom_model,
                    to_boom_matrix(self._prior.prior_transition_counts),
                    boom.GlobalRng.rng)
                self._boom_model.fix_pi0(self._boom_model.initial_distribution)
            else:
                self._boom_sampler = boom.MarkovConjugateSampler(
                    self._boom_model,
                    to_boom_matrix(self._prior.prior_transition_counts),
                    to_boom_vector(self._prior.prior_initial_counts),
                    boom.GlobalRng.rng)
            self._boom_model.set_method(self._boom_sampler)

        return self._boom_model

    def allocate_space(self, niter):
        dim = self.state_size
        self._transition_probability_draws = np.empty((niter, dim, dim))

    def record_draw(self, iteration):
        self._transition_probability_draws[iteration, :, :] = (
            self._boom_model.transition_probabilities.to_numpy())

    def create_boom_data_builder(self, data=None):
        from BayesBoom.R.boom_data_builders import (
            MarkovSufDataBuilder,
            LabelledMarkovDataBuilder,
            UnlabelledCategoricalDataBuilder,
        )
        if isinstance(data, list) and isinstance(data[0], MarkovSuf):
            return MarkovSufDataBuilder()
        elif self._categories:
            return LabelledMarkovDataBuilder(self._categories)
        else:
            return UnlabelledCategoricalDataBuilder(self.state_size)

    def sim(self, sample_size):
        P = self._transition_matrix
        states = np.arange(self.state_size)
        ind = np.empty(sample_size, dtype="int")
        rng = np.random.default_rng()
        ind[0] = rng.choice(a=states, size=1, p=self._initial_distribution)[0]
        for i in range(1, sample_size):
            ind[i] = rng.choice(a=states, size=1, p=P[ind[i - 1], :])[0]
        if self._categories is not None:
            return self._categories[ind]
        return ind

    def plot_components(self, components, burn=0, style="ts",
                        fig=None, ax=None, **kwargs):
        from BayesBoom.R.density import Density
        style = unique_match(style, ["ts", "density", "histogram", "boxplot"])
        K = self.state_size
        S = len(components)
        niter = self._transition_probability_draws.shape[0]
        if burn < 0:
            burn = 0

        probs = np.empty((niter - burn, S, K, K))
        for s in range(S):
            probs[:, s, :, :] = components[s]._transition_probability_draws[
                burn:, :, :]

        if fig is None:
            fig, ax = plt.subplots(K, K, sharex=True, sharey=True)
        else:
            ax = fig.subplots(K, K, sharex=True, sharey=True)

        draw = range(burn, niter)

        if style == "ts":
            for source in range(K):
                for dest in range(K):
                    for s in range(S):
                        ax[source, dest].plot(draw, probs[:, s, source, dest])
        elif style == "density":
            for source in range(K):
                for dest in range(K):
                    for s in range(S):
                        den = Density(probs[:, s, source, dest])
                        den.plot(ax=ax[source, dest])
        elif style == "boxplot":
            for source in range(K):
                for dest in range(K):
                    ax[source, dest].boxplot(probs[:, :, source, dest])
        else:
            raise Exception(f"Style argument '{style}' unrecognized.")

        return fig, ax
