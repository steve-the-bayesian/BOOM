import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
import copy

import matplotlib.pyplot as plt

from .base import unique_match

from .boom_py_utils import (
    to_boom_vector,
    to_boom_spd,
    to_boom_matrix,
)

from .plots import ensure_ax

from .boom_data_builders import (
    DataBuilder,
    IntDataBuilder,
    DoubleDataBuilder,
    VectorDataBuilder,
    LabelledCategoricalDataBuilder,
    UnlabelledCategoricalDataBuilder,
    LabelledMarkovDataBuilder,
    UnlabelledMarkovDataBuilder,
    MultilevelCategoricalDataBuilder,
)

from .density import Density



"""
Wrapper classes to encapsulate and expand models and prior distributions
from the Boom library.
"""

class DoubleModel(ABC):
    """
    A base class that marks its children as being able to produce a
    boom.DoubleModel, which is simply a model that implements a 'logp' method
    measuring a real valued random variable.
    """

    @abstractmethod
    def boom(self):
        """
        Return a boom.DoubleModel with parameters set from this object.
        """

    @property
    @abstractmethod
    def mean(self):
        """
        The mean of the distribution.
        """

    def create_boom_data_builder(self):
        return DoubleDataBuilder()

    def __eq__(self, other):
        return self.__dict__ == other.__dict__


class MixtureComponent(ABC):
    """
    Base class for models that can serve as mixture components.
    """

    @abstractmethod
    def allocate_space(self, niter):
        """
        Allocate space to store 'niter' MCMC draws of the model parameters.
        """

    @abstractmethod
    def record_draw(self, iteration):
        """
        Record the current model parameters (to be obtained from a stored
        boom model object) in position 'iteration' of previously allocated
        storage.
        """

    @abstractmethod
    def create_boom_data_builder(self):
        """
        Return a DataBuilder object that can convert Python data into the
        BOOM data of the format expected by the concrete (child) model.
        """

    @abstractmethod
    def plot_components(self, components, burn, style: str, fig, ax, **kwargs):
        """
        Args:
          components: A list of mixture components to plot.  It is expected that
            each component will be the same concrete class as the class
            implementing this function.  E.g. a NormalModel will implement
            plot_components expecting to be passed other NormalModel objects.
          burn: An integer giving the number of MCMC iterations to discard as
            burn-in.
          style: A string giving the style of plot to produce.  Common choices
            are:
            - "ts" a time series plot of MCMC iterations.
            - "den" for a density plot
            - "box" for boxplots
          fig: The matplotlib Figure object, which is used if the desired plot
            requires multiple axes.
          ax: A matplotlib Axes object, used if the desired plot can be plotted
            on a single set of axes.
          **kwargs: Keyword arguments passed to concrete plotting functions for
            child classes.
        
        Returns:
          The (fig, ax) pair on which the plot is drawn.
        """

    
class Ar1CoefficientPrior(DoubleModel):
    """
    Contains the information needed to create a prior distribution on an AR1
    coefficient.
    """
    def __init__(self,
                 mu: float = 0.0,
                 sigma: float = 1.0,
                 force_stationary: bool = True,
                 force_positive: bool = False,
                 initial_value: float = None):
        """
        Args:
          mu: The prior mean of the coefficient.
          sigma:  The prior standard deviation of the coefficient.
          force_stationary: If True then the prior support for the AR1
            coefficient will be truncated to (-1, 1).
          force_positive: If True then the prior for the AR1 coefficient will
            be truncated to positive values.
          initial_value: A suggestion about where to start an MCMC sampling
            run.  The default is to use mu.
        """
        self.mu = mu
        self.sigma = sigma
        self.force_stationary = force_stationary
        self.force_positive = force_positive
        self.initial_value = initial_value
        if initial_value is None:
            self.initial_value = mu

    def boom(self):
        """
        Return the boom.GaussianModel corresponding to this object's
        parameters.
        """
        import BayesBoom.boom as boom
        return boom.GaussianModel(self.mu, self.sigma)

    @property
    def mean(self):
        return self.mu

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, payload):
        self.__dict__ = payload


class BetaPrior(DoubleModel):
    """
    A distribution, typically used as the prior over a scalar probability.
    """
    def __init__(self, a=1.0, b=1.0):
        self._a = float(a)
        self._b = float(b)
        self._boom_model = None

    @property
    def mean(self):
        return self._a / (self._a + self._b)

    def boom(self):
        if self._boom_model is not None:
            return self._boom_model
        import BayesBoom.boom as boom
        self._boom_model = boom.BetaModel(self._a, self._b)
        return self._boom_model

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __getstate__(self):
        payload = self.__dict__
        payload["_boom_model"] = self._boom_model is not None
        return payload

    def __setstate__(self):
        self.__dict__ = payload
        if payload["_boom_model"]:
            self._boom_model = None
            self.boom()


class DirichletPrior:
    """
    A Dirichlet prior distribution over discrete probability distributions.
    """

    def __init__(self, counts):
        counts = np.array(counts)
        if not np.all(counts > 0):
            raise Exception("All elements of 'counts' must be positive.")
        self._counts = counts
        self._boom_model = None

    def boom(self):
        if self._boom_model is None:
            import BayesBoom.boom as boom
            self._boom_model = boom.DirichletModel(boom.Vector(self._counts))
        return self._boom_model

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __getstate__(self):
        ans = self.__dict__
        ans["_boom_model"] = self._boom_model is not None
        return ans

    def __setstate__(self, payload):
        self.__dict__ = payload
        if payload["_boom_model"]:
            self._boom_model = None
            self.boom()
    

class GammaModel(DoubleModel):
    def __init__(self, shape=None, scale=None, mu=None, a=None, b=None):
        """
        A GammaModel(a, b) can be defined either in terms of its shape (a) and
        scale (b) paramaeters (with mean a/b, variance a/b^2), or it's mean
        (mu) and shape parameters (so the mean is mu and the variance is
        mu^2/a).

        Args:
          shape:  The shape parameter a.
          scale:  The scale parameter b.
          mu:  The mean of the distribution.
          a:  Another name for the shape parameter.
          b:  Another name for the scale parameter.

        Only two of these parameters need to be specified.  If all three are
        given, then 'mu' is ignored.
        """
        if a is not None:
            shape = a
        if b is not None:
            scale = b

        if (shape is None) + (scale is None) + (mu is None) > 1:
            raise Exception("Two parameters must be specified.")

        self._a = shape
        if self._a is None:
            self._a = scale * mu

        self._b = scale
        if self._b is None:
            self._b = mu / shape

        if self._a <= 0 or self._b <= 0:
            raise Exception("GammaModel parameters must be positive.")

        self._boom_model = None

    @property
    def mean(self):
        self._refresh_params()
        return self._a / self._b

    @property
    def variance(self):
        self._refresh_params()
        return self._a / self._b**2

    @property
    def a(self):
        self._refresh_params()
        return self._a

    @property
    def shape(self):
        self._refresh_params()
        return self._a

    @property
    def b(self):
        self._refresh_params()
        return self._b

    @property
    def scale(self):
        self._refresh_params()
        return self._b

    def boom(self):
        if self._boom_model is None:
            import BayesBoom.boom as boom
            self._boom_model = boom.GammaModel(self._a, self._b)
        return self._boom_model

    def allocate_space(self, niter):
        self._a_draws = np.empty(niter)
        self._b_draws = np.empty(niter)

    def record_draw(self, iteration):
        if self._boom_model is not None:
            self._a_draws[iteration] = self._boom_model.alpha()
    
    def _refresh_params(self):
        if self._boom_model is not None:
            self._a = self._boom_model.a
            self._b = self._boom_model.b

    def __getstate__(self):
        payload = self.__dict__
        payload["_boom_model"] = self._boom_model is not None
        return payload

    def __setstate__(self, payload):
        self.__dict__ = payload
        if payload["_boom_model"]:
            self._boom_model = None
            self.boom()
            
    def __repr__(self):
        ans = f"A GammaModel with shape = {self.shape} "
        ans += f"and scale = {self.scale}."
        return ans


class GaussianSuf:
    """
    Sufficient statistics for a scalar normal model.
    """

    def __init__(self, data=None):
        """
        Args:
          data: If None (the default) then an empty GaussianSuf is created.
            Otherwise create a new GaussianSuf summarizing 'data'.
        """
        self._sum = 0
        self._sumsq = 0
        self._n = 0
        if data is not None:
            self.update(data)

    def update(self, incremental_data):
        """
        Add summaries of the incremental data to the data already summarized.

        Args:
          data:  A 1-d numpy array, or equivalent.

        Effects:
          The sufficient statistics in the object are updated to describe data.
        """
        y = np.array(incremental_data)
        self._sum += np.nansum(y)
        self._sumsq += np.nansum(y * y)
        self._n += np.sum(~np.isnan(incremental_data))

    def combine(self, other):
        """
        Add the sufficient statistics from 'other' to 'self'.  This operation is
        done inplace.  The 'other' object is unaffected.
        """
        self._n += other._n
        self._sum += other._sum
        self._sumsq += other._sumsq

    def __iadd__(self, other):
        """
        Implements operator +=.  Other can either be a GaussianSuf or raw data.
        """
        if isinstance(other, GaussianSuf):
            self.combine(other)
        else:
            self.update(other)
        return self

    def __add__(self, other):
        """
        Implements operator+.  Other can either be a GaussianSuf or raw data.
        """
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
        if self.sample_size > 0:
            return self._sum / self.sample_size
        else:
            return 0.0

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


class MarkovConjugatePrior:
    """
    Conjugate prior for a MarkovModel.  The parameters are
    'prior_transition_counts' and optionally 'prior_initial_counts'.
    """
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
    def __init__(self, categorical_data_sequence=None, levels=None, sort_levels=False):
        """
        Args:
          categorical_data_sequence: the sequence of categorical data from which
            to build the sufficient statistics.
          levels: The sequence of unique level values from which the entries in
            categorical_data_sequence take their values.  If None then the
            levels will be created from the unique entries in
            categorical_data_sequence.
          sort_levels: If True then the levels will be sorted alphabetically.
            If False then the levels appear in arbitrary order (likey the order
            they appear in categorical_data_sequence.)
        """

        # The most recent value to have been observed.
        self._last_value = None

        # The value observed at time zero0.
        self._initial_value = None

        # self._levels is a pd.Series indexed by the text of the level.  The
        # numeric value of the series is the
        
        # /arts_&_entertainment/comics_&_animation/anime_&_manga                  2391
        # /arts_&_entertainment/comics_&_animation/cartoons                        295
        # /arts_&_entertainment/entertainment_industry/film_&_tv_industry          454
        # /arts_&_entertainment/entertainment_industry/recording_industry        45594

        self._levels = self._create_levels(levels, categorical_data_sequence, sort_levels)

        # self._transition_counts is a data frame.  Rows and columns are indexed
        # by categorical levels.
        self._transition_counts = np.zeros((self.num_levels, self.num_levels))

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

    def top_transitions(self, n, omit_self_transitions=False, probs=False):
        """
        Return the top n transitions in the transition count matrix.

        Args:
          n:  The number of transitions desired.
          omit_self_transitions:  
        """
        trans = self.transition_counts.copy()
        if probs:
            row_totals = trans.sum(axis=1)
            row_totals[row_totals <= 0] = 1
            trans = trans.div(row_totals, axis="rows")
        
        if omit_self_transitions:
            np.fill_diagonal(trans.values, 0)
            
        return trans.stack().nlargest(n)

    def _create_levels(self, levels, categorical_data_sequence, sort_levels):
        """
        Return a pd.Series indexed by the levels of the 
        """
        if (levels is None) and (categorical_data_sequence is None):
            raise Exception("At least one of 'categorical_data_sequence' "
                            "or 'levels' is needed.")

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
            raise Exception("Sufficient statistics created with different levels.")
            
        ans = MarkovSuf(levels=self._levels, sort_levels=False)
        ans._transition_counts = self._transition_counts + other._transition_counts
        ans._initial_value_counts = self.initial_value_counts + other.initial_value_counts
        ans._last_value = other._last_value

        return ans

    def __iadd__(self, other):
        """
        Implements self += other
        """
        if np.any(self._levels != other._levels):
            raise Exception("Sufficient statistics created with different levels.")

        self._transition_counts += other._transition_counts
        self._initial_value_counts = self.initial_value_counts + other.initial_value_counts
        self._last_value = other._last_value

        return self
        
    
class MarkovModel(MixtureComponent):
    def __init__(self,
                 transition_matrix=None,
                 initial_distribution=None,
                 state_size=None,
                 categories=None):
        """
        Args:
          transition_matrix: A square matrix with element (r, s) giving the
            conditional probability of a transition to state s given current
            state r.  Each row sums to 1.  If None, then a matrix will be
            created giving uniform transition probability between any two
            states.
          state_size: An integer giving the number of states.  If None then
            transition_matrix must be supplied explicityly.  If
            transition_matrix is supplied this argument is not used.
          initial_distribution: A discrete probability distribution (as a numpy
            vector) giving the distribution of the state at time 0.  If None a
            uniform distribution is assumed for the initial state.
          categories: A sequence of strings describing the states of the Markov
            model.  This is an optional argument.  If it is used it will
            supercede 'state_size.'  If 'categories' is omitted then the states
            of the model will be integers 0, 1, 2, ..., state_size - 1.
        """
        if categories is not None:
            state_size = len(categories)
        
        if transition_matrix is None:
            if state_size is None:
                raise Exception("If transition_matrix is None then "
                                "state_size must be given.")
            transition_matrix = np.ones((state_size, state_size)) / state_size

        state_size = transition_matrix.shape[0]
        if categories and (state_size != len(categories)):
            raise Exception(
                f"The number of categories ({len(categories)}) does not match "
                f"the state size ({state_size}).")
        
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
                self._boom_model.transition_probabilities.to_numpy()
            )
        return self._transition_matrix

    @property
    def initial_distribution(self):
        if self._boom_model:
            self._initial_distribution = (
                self._boom_model.initial_distribution.to_numpy()
            )

        return self._initial_distribution

    def set_prior(self, prior):
        if not isinstance(prior, MarkovConjugatePrior):
            raise Exception("Prior must be of class MarkovConjugatePrior.")
        self._prior = prior
    
    def boom(self):
        if self._boom_model is not None:
            return self._boom_model

        import BayesBoom.boom as boom
        self._boom_model = boom.MarkovModel(
            to_boom_matrix(self._transition_matrix),
            to_boom_vector(self._initial_distribution))

        # if self._data is not None:
        #     # Add the data to the boom model
        #     pass
        
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
            self._boom_model.transition_probabilities.to_numpy()
        )

    def create_boom_data_builder(self):
        if self._categories:
            return LabelledMarkovDataBuilder(self._categories)
        else:
            return UnlabelledCategoricalDataBuilder(self._state_size)

    def sim(self, sample_size):
        P = self._transition_matrix
        states = np.arange(self.state_size)
        ind = np.empty(sample_size, dtype="int")
        rng = np.random.default_rng()
        ind[0] = rng.choice(a=states, size=1, p=self._initial_distribution)[0]
        for i in range(1, sample_size):
            ind[i] = rng.choice(a=states, size=1, p=P[ind[i-1], :])[0]

        if self._categories is not None:
            return self._categories[ind]
        else:
            return ind

    def plot_components(self, components, burn=0, style="ts",
                        fig=None, ax=None, **kwargs):
        """
        A KxK array of plots, where K is self.state_size.  Each plot has S
        entries, where S is the number of mixture components.
        """

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
            # A KxK array of time series plots.
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
            raise Exception(f"Style argument '{style}' unrecognized. ")

        return fig, ax


class MultilevelMultinomialModel(MixtureComponent):
    def __init__(self, taxonomy, sep="/"):
        """
        Args:
          taxonomy: A sequence of strings giving taxonomy id's in the form
            "top/second/third/etc".  The number of levels in each entry can
            vary.  Each level of the taxonomy that has children will be assigned
            a sub-model.
        """
        import BayesBoom.boom as boom        
        self._sep = sep
        if isinstance(taxonomy, boom.Taxonomy):
            self._boom_taxonomy = taxonomy
        else:
            self._boom_taxonomy = boom.Taxonomy(taxonomy, sep)
        self._boom_model = None
        self._boom_sampler = None

        # A list of strings containing all parent levels of the taxonomy.
        self._model_levels = None

        # A dict, keyed by the entries in self._model_levels, with each entry
        # containing a np.ndarray containing the draws for the conditional model
        # probabilities for a specific taxonomy parent level.
        self._draws = None

    def probs(self, level=""):
        if self._boom_model is None:
            self.boom()
        
        if level:
            return self._boom_model.conditional_model(
                level, self._sep).probs.to_numpy()
        else:
            return self._boom_model.top_level_model.probs.to_numpy()

    def prob_draws(self, parent_level="", conditional=False, burn=0):
        """
        Args:
          parent_level: The set of probabilities immediately underneath this
            level of the taxonomy.  If 'parent_level' is the empty string then
            the probs are the first level of the taxonomy.
          conditional: If True then the returned probabilities sum to 1 within
            each draw.  These are the conditional model probabilities of the
            taxonomy child-levels conditional on the parent level occurring.
 
        Returns:
          A pd.DataFrame containing the set of model probabilities simulated
          from the posterior distribution for the desired level.  The column
          headings of the data frame are the child levels of the supplied parent
          level.
        """
        levels = self._boom_taxonomy.child_levels(parent_level)
        draws = pd.DataFrame(self._draws[parent_level],
                             columns=levels)

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
        if (self._boom_model is None):
            self._boom_model = boom.MultilevelMultinomialModel(
                self._boom_taxonomy)
            self._ensure_posterior_sampler();

        # A list of strings containing all parent levels of the taxonomy.
        self._model_levels = self._boom_model.model_levels

        return self._boom_model

    def allocate_space(self, niter):
        self.boom()

        # params is a list of boom Vector's containing the conditional
        # probabilities at each parent level of the taxonomy.  We need it here
        # to get the parameter sizes.
        params = self._boom_model.parameters

        # params and self._model_levels are stored in the same order.
        self._draws = {
            x[0]: np.empty((niter, x[1].size))
            for x in zip(self._model_levels, params)
        }

    def record_draw(self, iteration):
        for draw in zip(self._model_levels, self._boom_model.parameters):
            self._draws[draw[0]][iteration, :] = draw[1].to_numpy()
        
    def create_boom_data_builder(self):
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
        S = len(components)
        if burn < 0:
            burn = 0

        niter = self._draws[self._model_levels[0]].shape[0]
        levels = self._boom_taxonomy.child_levels(parent_level)
        K = len(levels)
        draws = np.empty((niter - burn, S, K))
        for s in range(S):
            draws[:, s, :] = components[s].prob_draws(
                burn=burn,
                parent_level=parent_level,
                conditional=conditional)

        style = unique_match(
            style,
            ["ts", "boxplot", "barplot", "density"]
        )

        if (style == "ts") or (style == "boxplot") or (style == "density"):
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
            # mean_probs is an S x K matrix with rows summing to 1.
            
            states = np.arange(S)
            cum_probs = np.zeros(S)
            grays = np.arange(K) / K

            for k in range(K):
                ax.bar(states,
                       mean_probs[:, k],
                       bottom=cum_probs,
                       width=.6,
                       color=str(grays[k]))
                cum_probs += mean_probs[:, k]

            if S < 10:
                ax.set_xticks(np.arange(S),
                              [str(s) for s in range(S)])
        else:
            raise Exception(f"Style {style} unrecognized.")

        return fig, ax
    
    def _ensure_posterior_sampler(self):
        import BayesBoom.boom as boom
        self._boom_sampler = boom.MultilevelMultinomialPosteriorSampler(
            self._boom_model,
            1.0)  ####### TODO: add prior control
        self._boom_model.set_method(self._boom_sampler)


class MultinomialModel(MixtureComponent):
    """
    A Python wrapper for the boom MultinomialModel object.
    """

    def __init__(self, probs, categories=None):
        """
        Args:
          probs: A numpy vector of probabilities.  This is a discrete
            probability distribution: nonnegative values summing to 1.
          categories: Optional numpy vector of category names.  If supplied, the
            vector's lengh must match probs.
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
        else:
            return len(self._probs)
        
    def allocate_space(self, niter):
        self._prob_draws = np.empty((niter, self.dim))

    def record_draw(self, iteration):
        if not self._boom_model:
            raise Exception("Object contains no boom model")
        self._prob_draws[iteration, :] = self.probs

    def set_prior(self, prior):
        self._prior = prior
        
    def boom(self):
        import BayesBoom.boom as boom
        if self._boom_model:
            return self._boom_model
        
        if not self._boom_model:
            self._boom_model = boom.MultinomialModel(
                to_boom_vector(self._probs))

        if self._prior:
            if isinstance(self._prior, DirichletPrior):
                self._boom_sampler = boom.MultinomialDirichletSampler(
                    self._boom_model,
                    self._prior.boom(),
                    boom.GlobalRng.rng)
            else:
                raise Exception("Not sure how to create a posterior sampler.")

            self._boom_model.set_method(self._boom_sampler)
                
        return self._boom_model

    def sim(self, sample_size = 1):
        rng = np.random.default_rng()
        values = rng.choice(a=self._categories, size=sample_size, p=self._probs)
        return values
        
    def create_boom_data_builder(self):
        if isinstance(self._categories, range):
            return UnlabelledCategoricalDataBuilder(self.dim)
        else:
            return LabelledCategoricalDataBuilder(self._categories)

    def plot_components(self,
                        components,
                        burn=0,
                        style="ts",
                        levels=None,
                        fig=None,
                        ax=None,
                        **kwargs):
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
            local_probs = components[s]._prob_draws[burn:, :]
            probs[:, s, :] = np.array(local_probs)

        style = unique_match(
            style,
            ["ts", "boxplot", "barplot", "density"]
        )

        if style == "ts" or style == "boxplot" or style == "density":
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
                                ax[i, j].plot(
                                    iteration,
                                    probs[:, s, counter],
                                    label=str(s))
                        elif style == "boxplot":
                            ax[i, j].boxplot(probs[:, :, counter])
                        elif style == "density":
                            for s in range(S):
                                den = Density(probs[:, s, counter])
                                den.plot(ax=ax, label=str(s))
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
            # mean_probs is an S x K matrix with rows summing to 1.
            
            states = np.arange(S)
            cum_probs = np.zeros(S)
            grays = np.arange(K) / K

            for k in range(K):
                ax.bar(states,
                       mean_probs[:, k],
                       bottom=cum_probs,
                       width=.6,
                       color=str(grays[k]))
                cum_probs += mean_probs[:, k]

            if S < 10:
                ax.set_xticks(np.arange(S),
                              [str(s) for s in range(S)])

        else:
            raise Exception(f"Style argument {style} unrecognized. ")
        

    
class MvnBase(ABC):
    @property
    @abstractmethod
    def dim(self):
        """
        The dimension of the variable described by the distribution.
        """
    @property
    @abstractmethod
    def mean(self):
        """
        The mean of the distribution.
        """

    @property
    @abstractmethod
    def variance(self):
        """
        The variance of the distribution, as a 2-d numpy array.
        """

    @abstractmethod
    def boom(self):
        """
        Return the corresponding boom object.
        """

    @property
    def create_boom_data_builder(self):
        return VectorDataBuilder()


class MvnPrior(MvnBase):
    """
    Encodes a multivariate normal distribution.
    """
    def __init__(self, mu, Sigma):
        if len(mu.shape) != 1:
            raise Exception("mu must be a vector.")
        if len(Sigma.shape) != 2:
            raise Exception("Sigma must be a matrix.")
        if Sigma.shape[0] != Sigma.shape[1]:
            raise Exception("Sigma must be symmetric")
        if Sigma.shape[0] != len(mu):
            raise Exception("mu and Sigma must be the same dimension.")
        self._mu = mu
        self._Sigma = Sigma

    @property
    def dim(self):
        return len(self._mu)

    @property
    def mu(self):
        return self._mu

    @property
    def mean(self):
        return self.mu

    @property
    def Sigma(self):
        return self._Sigma

    @property
    def variance(self):
        return self.Sigma

    def boom(self):
        """
        Return the boom.MvnModel corresponding to this object's parameters.
        """
        if self._boom_model is None:
            import BayesBoom.boom as boom
            self._boom_model = boom.MvnModel(boom.Vector(self._mu),
                                             boom.SpdMatrix(self._Sigma))
        return self._boom_model


class MvnGivenSigma(MvnBase):
    """
    Encodes a conditional multivariate normal distribution given an external
    variance matrix Sigma.  This model describes y ~ Mvn(mu, Sigma / kappa).
    """
    def __init__(self, mu: np.ndarray, sample_size: float):
        self._mu = np.array(mu, dtype="float").ravel()
        self._sample_size = float(sample_size)
        self._boom_model = None

    @property
    def dim(self):
        return len(self._mu)

    def boom(self):
        if self._boom_model is None:
            import BayesBoom.boom as boom
            self._boom_model = boom.MvnGivenSigma(to_boom_vector(self._mu),
                                                  self._sample_size)
        return self._boom_model

    @property
    def variance(self):
        raise Exception("MvnGivenSigma needs Sigma value to compute the "
                        "variance.")

    @property
    def mean(self):
        return self._mu

    def __eq__(self, other):
        return self.__dict__ == other.__dict__


class NormalPrior(DoubleModel, MixtureComponent):
    """
    A scalar normal prior distribution.
    """
    def __init__(self,
                 mu: float = 0.0,
                 sigma: float = 1.0,
                 initial_value: float = None):
        self.mu = float(mu)
        self.sigma = float(sigma)
        if initial_value is None:
            self.initial_value = mu
        else:
            self.initial_value = float(initial_value)

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
        elif isinstance(prior, NormalPrior):
            self._mean_prior = prior
        elif isinstance(prior, GammaModelBase):
            self._sd_prior = prior
    
    def boom(self):
        """
        Return the boom.GaussianModel corresponding to the object's parameters.
        """
        import BayesBoom.boom as boom
        if self._boom_model is None:
            import BayesBoom.boom as boom
            self._boom_model = boom.GaussianModel(self.mu, self.sigma)

            if (
                    (self._prior is not None)
                    and
                    isinstance(self._prior, NormalInverseGammaModel)
            ):
                boom_sampler = boom.GaussianConjugateSampler(
                    self._boom_model,
                    self._prior.gaussian_given_sigma(self._boom_model.sigsq_parameter),
                    self._prior.chisq())
                self._boom_model.set_method(boom_sampler)

            else:
                raise Exception("Not sure how to build the posterior sampler.")
                
        return self._boom_model

    def allocate_space(self, niter):
        self._mu_draws = np.empty(niter)
        self._sigma_draws = np.empty(niter)

    def record_draw(self, iteration):
        self._mu_draws[iteration] = self._boom_model.mu
        self._sigma_draws[iteration] = self._boom_model.sigma
    
    def create_boom_data_builder(self):
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
            fig, ax = plt.subplots(1,numplots)
        elif ax is None:
            ax = fig.subplots(1,numplots)

        if params == "both":
            self._plot_params(components=components,
                              burn=burn,
                              style=style,
                              fig=fig,
                              ax=ax[0],
                              what="means",
                              **kwargs)
            self._plot_params(components=components,
                              burn=burn,
                              style=style,
                              fig=fig,
                              ax=ax[1],
                              what="sds",
                              **kwargs)

        elif params == "mean":
            self._plot_params(components=components,
                              burn=burn,
                              style=style,
                              fig=fig,
                              ax=ax,
                              what="means",
                              **kwargs)

        elif params == "sd":
            self._plot_params(components=components,
                              burn=burn,
                              style=style,
                              fig=fig,
                              ax=ax,
                              what="sds",
                              **kwargs)

        else:
            raise Exception(f"Unknown value of params: {params}")
        
        return fig, ax

    def _plot_params(self,
                     components,
                     burn=0,
                     style="ts",
                     fig=None,
                     ax=None,
                     what = "means",
                     **kwargs):
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
        payload = {
            "mu": self.mean,
            "sigma": self.sd,
            "initial_value": self.initial_value,
        }
        return payload

    def __setstate__(self, payload):
        self.mu = payload["mu"]
        self.sigma = payload["sigma"]
        self.initial_value = payload["initial_value"]


class NormalInverseGammaModel:
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
        return self._sd_estimate**2 * self._df
        
    def gaussian_given_sigma(self, sigsq_parameter):
        """
        Returns a BOOM GaussianModelGivenSigma model object.
        """
        import BayesBoom.boom as boom
        return boom.GaussianModelGivenSigma(
            sigsq_parameter,
            self._mean,
            self._mean_prior_sample_size)

    def chisq(self):
        import BayesBoom.boom as boom
        return boom.ChisqModel(
            self._df,
            self._sd_estimate)
        
        
class PoissonModel(MixtureComponent):
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
    
    def boom(self):
        if self._boom_model is not None:
            return self._boom_model
        
        import BayesBoom.boom as boom
        self._boom_model = boom.PoissonModel(self._lambda)
        if self._prior is not None:
            if isinstance(self._prior, GammaModel):
                self._boom_sampler = boom.PoissonGammaSampler(
                    self._boom_model,
                    self._prior.boom(),
                    boom.GlobalRng.rng)
            else:
                raise Exception(
                    f"PoissonModel has a prior of class {type(self._prior)}.  "
                    "I'm unclear how to create a sampler.")
                    
            self._boom_model.set_method(self._boom_sampler)

        return self._boom_model

    def allocate_space(self, niter):
        self._lambda_draws = np.empty(niter)

    def record_draw(self, iteration):
        self._lambda_draws[iteration] = self._boom_model.mean

    def create_boom_data_builder(self):
        return IntDataBuilder()

    def plot_components(self,
                        components,
                        burn=0,
                        style="ts",
                        fig=None,
                        ax=None,
                        **kwargs):
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
                ax.plot(iteration, draws[:, s], label = str(s))
            ax.legend(loc="upper right")

        elif style == "density":
            for s in range(S):
                den = Density(draws[:, s])
                den.plot(ax=ax, label = str(s))
            ax.legend(loc = "upper right")

        elif style == "boxplot":
            ax.boxplot(draws)

        else:
            raise Exception(f"Style argument '{style}' unrecognized. ")

        return fig, ax
    
        
class RegSuf:
    """
    The sufficient statistics needed to specify a regression model.
    """

    def __init__(self, xtx, xty, sample_sd, sample_size=None, ybar=None,
                 xbar=None):
        """
        In what follows X is the design matrix of predictors, and y is the
        column vector of responses.  The matrix transpose of X is denoted X'.

        Args:
          xtx: The cross product matrix X'X.
          xty: X'y
          sample_sd:  The sample standard deviation of the y's.
          sample_size: The number of observations covered by the sufficient
            statistics.  If X contains a column of 1's in column 0 then
            sample_size can be None.

          ybar: The mean of the y's (a scalar).  If X contains a column of 1's
            in column 0 then this can be None.

          xbar: The mean of the X's (a vector).
        """
        xtx = np.array(xtx)
        xty = np.array(xty)
        xbar = np.array(xbar)

        if xtx.shape[0] != xtx.shape[1]:
            raise Exception("xtx must be square")
        if xtx.shape[0] != xty.shape[0]:
            raise Exception("xtx and xty must be the same size.")

        if not sample_sd >= 0:
            raise Exception("The sample_sd must be non-negative.")

        if sample_size is None:
            sample_size = xtx[0, 0]
        if not sample_size >= 0:
            raise Exception("The sample size must be non-negative.")

        if xbar is None:
            raise Exception("xbar must be supplied.")
        if xbar.shape[0] != xty.shape[0]:
            raise Exception("xbar has the wrong size.")

        if ybar is None:
            ybar = xty[0] / sample_size

        self._xtx = xtx
        self._xty = xty
        self._sample_sd = sample_sd
        self._sample_size = sample_size
        self._ybar = ybar
        self._xbar = xbar

    @classmethod
    def from_data(cls, X, y):
        X = np.array(X)
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        y = np.array(y).ravel()

        if X.shape[0] != y.shape[0]:
            raise Exception(
                f"The length of y ({len(y)}) must match the number of rows "
                f"in X ({X.shape[0]}).")

        xtx = X.T @ X
        xty = X.T @ y
        sample_size = len(y)
        sample_sd = np.std(y, ddof=1)
        ybar = np.mean(y)
        xbar = X.mean(axis=0)
        return cls(xtx, xty, sample_sd, sample_size, ybar, xbar)

    def boom(self):
        import BayesBoom.boom as boom
        import BayesBoom.R as R
        return boom.RegSuf(xtx=to_boom_spd(self._xtx),
                           xty=to_boom_vector(self._xty),
                           sample_sd=self._sample_sd,
                           sample_size=self._sample_size,
                           ybar=self._ybar,
                           xbar=self._xbar)

    @property
    def xtx(self):
        return self._xtx

    @property
    def xty(self):
        return self._xty

    @property
    def xdim(self):
        return self._xtx.shape[0]

    @property
    def xbar(self):
        return self._xbar

    @property
    def mean_x(self):
        return self._xbar

    @property
    def mean_y(self):
        return self._ybar

    @property
    def ybar(self):
        return self._ybar

    @property
    def sample_sd(self):
        return self._sample_sd

    @property
    def sample_variance(self):
        return self._sample_sd**2

    @property
    def sample_size(self):
        return self._sample_size


class ScottZellnerMvnPrior(MvnBase):
    """
    A Zellner prior on a set of regression coefficients, shrunk towards the
    diagonal by a parameterized amount.

    The Zellner prior is a multivariate normal distribution with mean mu and
    precision matrix A = g * X'X / sigsq, where g is a number specified by the
    modeler, and sigsq is the residual variance parameter in the regression
    model.

    The ScottZellnerPrior is a modified version of the Zellner prior with X'X
    replaced by (1 - a) * X'X + a * diag(X'X).  That is, a weighted average of
    X'X with its diagonal.

    The coefficient 'g' in the ordinary Zellner prior is replaced by
    'prior_nobs' / n where n is the sample size.  Because X'X is the total
    information from the data available in a regression problem, X'X/n is the
    average information available from a single data point.  Thus 'prior_nobs'
    can be interpreted as the number of data points worth of information you
    want the prior to weigh.

    """

    def __init__(self,
                 suf: RegSuf,
                 diagonal_shrinkage: float = .05,
                 prior_nobs: float = 1.0,
                 sigma: float = 1.0):
        """
        Args:
          suf: The sufficient statistics of the regression model.
          diagonal_shrinkage: The 'a' parameter in the class description.  The
            amount by which to shrink towards the diagonal of X'X.  A real
            number between 0 and 1.
          prior_nobs: The number of observations worth of weight to assign the
            prior.  A positive scalar.
          sigma: A scale factor for the prior.  In some applications sigma must
            be determined outside this class.  Child classes may obtain sigma
            from a callback, for example.
        """
        omega = suf.xtx
        weight = diagonal_shrinkage
        omega = (1 - weight) * omega + weight * np.diag(np.diag(omega))

        omega = omega * (prior_nobs / suf.sample_size)
        self._precision = omega / sigma**2

        self._mean = np.zeros(omega.shape[0])
        self._mean[0] = suf.ybar

        self._variance = None

    @property
    def dim(self):
        return self._mean.shape[0]

    @property
    def mean(self):
        return self._mean

    @property
    def variance(self):
        if self._variance is None:
            self._variance = np.linalg.inv(self._precision)
        return self._variance

    def boom(self):
        import BayesBoom.boom as boom
        return boom.MvnModel(to_boom_vector(self.mean),
                             to_boom_spd(self.variance))


class SdPrior(DoubleModel):
    """
    A prior distribution for a standard deviation 'sigma'.  This prior
    assumes that 1/sigma**2 ~ Gamma(a, b), where a = df/2 and b = ss/2.  Here
    'df' is the 'sample_size' and ss is the "sum of squares" equal to the sample
    size times 'sigma_guess'**2.

    This prior allows an upper limit on the support of sigma, which is infinite
    by default.
    """

    def __init__(self, sigma_guess, sample_size=.01, initial_value=None,
                 fixed=False, upper_limit=np.inf):
        """
        Create an SdPrior.

        Args:
          sigma_guess:  Guess at the value of the standard deviation.
          sample_size: Number of observations worth of information with which
            to weight the guess.
          initial_value: The initial value to be used in an MCMC chain.  This
            is not always respected.  The default value is sigma_guess.
          fixed: Flag indicating whether the parameter should be held fixed in
            an MCMC algorithm.  This is mainly for debugging and is not always
            respected.
          upper_limit: Upper limit on the value of 'sigma'.
        """
        self.sigma_guess = float(sigma_guess)
        self.sample_size = float(sample_size)
        if initial_value is None:
            initial_value = sigma_guess
        self.initial_value = float(initial_value)
        self.fixed = bool(fixed)
        self.upper_limit = float(upper_limit)

    @property
    def sum_of_squares(self):
        return self.sigma_guess**2 * self.sample_size

    def create_chisq_model(self):
        return self.boom()

    def boom(self):
        """
        Return the boom.ChisqModel corresponding to the input parameters.
        """
        import BayesBoom.boom as boom
        return boom.ChisqModel(self.sample_size, self.sigma_guess)

    @property
    def mean(self):
        """
        The mean of the distribution on the precision scale.
        """
        return self.sample_size / self.sigma_guess**2

    def __repr__(self):
        ans = f"SdPrior with sigma_guess = {self.sigma_guess}, "
        ans += f"sample_size = {self.sample_size}, "
        ans += f"upper_limit = {self.upper_limit}"
        return ans

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, payload):
        self.__dict__ = payload


class UniformPrior(DoubleModel):
    """
    Univariate uniform distribution.
    """
    def __init__(self, lo, hi):
        if hi < lo:
            lo, hi = hi, lo
        self._lo = lo
        self._hi = hi

    @property
    def mean(self):
        return .5 * (self._lo + self._hi)

    def boom(self):
        """
        Return the boom.UniformModel corresponding to this object's parameters.
        """
        import BayesBoom.boom as boom
        return boom.UniformModel(self._lo, self._hi)

    
class WishartPrior:
    def __init__(self, df: float, variance_estimate: np.ndarray):
        """
        Args:
          df: The prior sample size.  For the distribution to be proper df must
            be larger than the number of rows in 'variance_estimate'.
          variance_estimate: A symmetric positive definite matrix defining the
            center of the distribution.

        Let X_i ~ Mvn(0, V).  Then the Wishart(nu, V) distribution describes
        the sum of 'nu' draws X_i * X_i'.  If the draws are placed as rows in a
        matrix X then X'X ~ Wishart(nu, V).  The mean of this distribution is
        nu * V.

        The Wishart distribution is the conjugate prior for the precision
        parameter (inverse variance) of the multivariate normal distribution.
        """
        sumsq = df * variance_estimate
        if len(sumsq.shape) != 2:
            raise Exception("sumsq must be a matrix")

        if sumsq.shape[0] != sumsq.shape[1]:
            raise Exception("sumsq must be square")

        sym_sumsq = (sumsq + sumsq.T) * .5
        sumabs = np.sum(np.abs(sumsq - sym_sumsq))
        relative = np.sum(np.abs(sumsq))
        if sumabs / relative > 1e-8:
            raise Exception("sumsq must be symmetric")

        if df <= sumsq.shape[0]:
            raise Exception(
                "df must be largern than nrow(sumsq) for the prior to be "
                "proper.")

        self._df = df
        self._sumsq = sumsq

    @property
    def variance_estimate(self):
        return self._sumsq / self._df

    @property
    def df(self):
        return self._df

    def boom(self):
        import BayesBoom.boom as boom
        return boom.WishartModel(self.df, self.variance_estimate)


