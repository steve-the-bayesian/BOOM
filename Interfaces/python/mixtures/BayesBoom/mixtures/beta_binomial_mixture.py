import BayesBoom.boom as boom
import pandas as pd
import numpy as np
import BayesBoom.R as R
from scipy.stats import betabinom


class BetaBinomialMixture:
    """
    A finite mixture of BetaBinomial distributions.

    Example use:

    # Create an empty model object.
    model = BetaBinomialMixture()

    # To fit a 3 component mixture, add three components.
    model.add_component(R.BetaPrior(1.0, 1.0), R.UniformPrior(0.1, 1000.0))
    model.add_component(R.BetaPrior(1.0, 1.0), R.UniformPrior(0.1, 1000.0))
    model.add_component(R.BetaPrior(1.0, 1.0), R.UniformPrior(0.1, 1000.0))

    # Add the data.
    model.add_data(my_three_column_matrix)

    # Fit the model by running MCMC for 'niter' iterations.
    model.mcmc(niter=1000)

    """

    def __init__(self, use_data_augmentation=False):
        self._components = list()
        self._mixing_distribution_prior_counts = list()
        self._boom_model = None
        self._use_data_augmentation = use_data_augmentation

    def add_component(self,
                      mean_prior: R.BetaPrior,
                      sample_size_prior: R.DoubleModel,
                      prior_count: float = 1.0):
        """
        Add a beta-binomial mixture component to the model.  The beta-binomial
        distribution is parameterized by two parameters: a and b, which can be
        thought of as counts of successes (a) and failures (b) in a binomial
        inference problem.  The mean of the distribution is a/(a+b), and the
        variance is controlled by the "sample size" a+b.

        Args:
          mean_prior: A prior distribution on the mean (a/a+b) of the component.
          sample_size_prior: A prior distribution on the sample size (a+b) of
            the component.
          prior_count: The mixing distribution has a dirichlet prior, which is
            parameterized by a vector of positive real numbers interpretable as
            prior counts.  This argument is the prior count for the frequency
            of this component in the mixture.
        """
        self._mixing_distribution_prior_counts.append(prior_count)
        self._mixing_weights = None
        component = {
            "mean_prior": mean_prior,
            "sample_size_prior": sample_size_prior
        }
        self._components.append(component)

    def add_data(self, data):
        """
        data: A three-column matrix of numbers giving the (a) number of trials,
          (b) the number of successes, and (c) the number of cases with that
          many trials and successes.

        Example:
        10, 4, 28   # There were 28 cases of 10 trials showing 4 successes.
         5, 0, 2    # There were 2 cases of 5 trials showing no successes.
        """
        if self._boom_model is None:
            self._boom_model = self._build_boom_model()
        if isinstance(data, pd.DataFrame):
            data = data.values
        self._boom_model.add_data(R.to_boom_matrix(data.astype(float)))

    def mcmc(self, niter, ping=None, seed=None):
        """
        Run the MCMC algorithm for 'niter' iterations.

        Args:
          niter:  The desired number of MCMC iterations.
          ping:  The frequency with which to print status updates.
          seed:  The random seed to use for the C++ random number generator.

        Returns:
          None

        Effects:
          MCMC draws are stored in the object.
        """
        if self._boom_model is None:
            self._boom_model = self._build_boom_model()
            import warnings
            warnings.warn("Running MCMC on a model with no data assigned.")

        self._create_storage(niter)
        if seed is not None:
            # I don't have a lot of confidence that this works.
            boom.GlobalRng.rng.seed(int(seed))
        for i in range(niter):
            R.report_progress(i, ping)
            self._boom_model.sample_posterior()
            self._record_draw(i)

    @property
    def number_of_mixture_components(self):
        """
        The number of mixture components contained in the object.
        """
        return len(self._mixing_distribution_prior_counts)

    @property
    def niter(self):
        """
        The number of MCMC iterations that have been run.
        """
        if self._mixing_weights is None:
            return 0
        return self._mixing_weights.shape[0]

    @property
    def a(self):
        """
        MCMC draws of the 'success count' parameters.  This is a matrix with
        'niter' rows and 'number_of_mixture_components' columns, with each
        column representing the 'a' parameter of one mixture component.
        """
        ans = np.empty((self.niter, self.number_of_mixture_components))
        for i in range(self.number_of_mixture_components):
            ans[:, i] = self._components[i]["draws"][:, 0]
        return ans

    @property
    def b(self):
        """
        MCMC draws of the 'failure count' parameters.  This is a matrix with
        'niter' rows and 'number_of_mixture_components' columns, with each
        column representing the 'b' parameter of one mixture component.
        """
        ans = np.empty((self.niter, self.number_of_mixture_components))
        for i in range(self.number_of_mixture_components):
            ans[:, i] = self._components[i]["draws"][:, 1]
        return ans

    @property
    def means(self):
        """
        MCMC draws of the mean parameter for each mixture component.
        """
        a = self.a
        b = self.b
        return a / (a + b)

    @property
    def sample_sizes(self):
        """
        MCMC draws of the sample size parameter for each mixture component.
        """
        return self.a + self.b

    @property
    def mixing_weights(self):
        """
        MCMC draws of the mixing weights.
        """
        return self._mixing_weights

    def restore_params(self, i: int):
        """
        Restore the parameters of the underlying boom model to iteration i.

        Note that i can be any legal python index.  The value -1 is particularly
        useful because it restores the parameters to the most recent draw.
        """
        mixing_weights = self._mixing_weights[i, :]
        self._boom_model.mixing_distribution.set_probs(
            R.to_boom_vector(mixing_weights))

        for comp in range(self.number_of_mixture_components):
            ab = self._components[comp]["draws"][i, :]
            self._boom_model.mixture_component(comp).set_a(ab[0])
            self._boom_model.mixture_component(comp).set_b(ab[1])

    def cluster_membership_probabilities(
            self, trials: int, successes: int, burn: int = 0):
        """
        Return the cluster membership probabilities for a user with the given
        number of successes and trials.

        Args:
          trials:  A number of binomial trials.
          successes:  A success count 0 <= successes <= trials.
          burn:  The number of MCMC iterations to discard as burn-in.

        Returns:
          A vector of cluster membership probabilities.
        """
        from .utils import normalize_logprob

        log_probs = np.log(self._mixing_weights)
        a = self.a
        b = self.b
        if burn is not None and burn > 0:
            a = a[burn:, :]
            b = b[burn:, :]
            log_probs = log_probs[burn:, :]

        for component in range(log_probs.shape[1]):
            log_probs[:, component] += betabinom.logpmf(
                successes, trials, a[:, component], b[:, component])

        probs = normalize_logprob(log_probs)
        return probs.mean(axis=0)

    def density_distribution(self,
                             trials: int,
                             burn: int = 0):
        """
        Args:
          trials: The number of trials ("N") on which to condition the
            distribution.
          burn:  The number of MCMC interations to discard as burn-in.


        Returns:
          A dict with the following keys:
          - "density": A matrix containing density values.  Rows correspond to
              MCMC iteration, and columns correspond to the numbers 0, 1, 2,
              ..., trials.
          - "weighted_components": A 3-way array of weighted density values.
            Element [i, j, k] contains MCMC draw i of the weighted density value
            of component j evaluated at k.
          - "unweighted_components": Same dimension as 'weighted_components' but
            the mixture components are not multiplied by the mixing weights.
        """
        a = self.a
        b = self.b
        w = self._mixing_weights
        if burn > 0:
            a = a[burn:, :]
            b = b[burn:, :]
            w = w[burn:, :]

        niter = a.shape[0]

        weighted_components = np.empty(
            (niter, self.number_of_mixture_components, trials + 1))
        unweighted_components = np.empty(
            (niter, self.number_of_mixture_components, trials + 1))

        for y in range(trials+1):
            for s in range(self.number_of_mixture_components):
                component = betabinom.pmf(y, trials, a[:, s], b[:, s])
                weighted_components[:, s, y] = w[:, s] * component
                unweighted_components[:, s, y] = component.copy()

        return {
            "density": np.sum(weighted_components, axis=1),
            "weighted_components": weighted_components,
            "unweighted_components": unweighted_components,
        }

    def plot_components(self,
                        trials: int,
                        burn: int = 0,
                        weighted: bool = True,
                        ax=None):
        """
        Plot the mixture components on a matplotlib.Axes object.

        Args:
          trials:  The number of trials on which to condition the distribution.
          burn:  The number of MCMC iterations to discard as burn-in.
          weighted: Should the weighted (True) or unweighted (False) mixture
            components be plotted.
          ax: The Axes object on which to draw the plot.  If None then a new
            Axes object will be created, and plt.show() will be called.

        Returns:
          The Axes object containing the graph, and the posterior means of the
          component densities being plotted.
        """

        import matplotlib.pyplot as plt

        component_distribution = self.density_distribution(
            trials=trials, burn=burn)

        if weighted:
            component_means = np.mean(component_distribution[
                "weighted_components"], axis=0)
        else:
            component_means = np.mean(component_distribution[
                "unweighted_components"], axis=0)

        if ax is None:
            _, ax = plt.subplots(1, 1)
            call_show = True
        else:
            call_show = False

        x = np.arange(trials + 1)
        colors = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']

        dx = 0.5 / self.number_of_mixture_components
        for s in range(self.number_of_mixture_components):
            ax.bar(x - .25 + s*dx,
                   component_means[s, :],
                   color=colors[s % len(colors)],
                   width=dx,
                   label="Component " + str(s))
        ax.legend()

        if call_show:
            plt.show()

        return ax, component_means

    def _create_storage(self, niter: int):
        """
        Create storage for 'niter' MCMC draws.
        """
        for component in self._components:
            component["draws"] = np.empty((niter, 2))
        self._mixing_weights = np.empty(
            (niter, self.number_of_mixture_components))

    def _record_draw(self, i):
        for c, component in enumerate(self._components):
            component["draws"][i, :] = [
                self._boom_model.mixture_component(c).a,
                self._boom_model.mixture_component(c).b
            ]
        self._mixing_weights[i, :] = (
            self._boom_model.mixing_distribution.probs.to_numpy()
        )

    def _build_boom_model(self):
        """
        Build a boom_model from
        """
        prior_counts = np.array(self._mixing_distribution_prior_counts)
        prior_mixing_distribution = prior_counts / np.sum(prior_counts)

        mixing_distribution = boom.MultinomialModel(
            R.to_boom_vector(prior_mixing_distribution))
        dirichlet_prior = boom.DirichletModel(R.to_boom_vector(prior_counts))

        mixing_distribution_sampler = boom.MultinomialDirichletSampler(
            mixing_distribution, dirichlet_prior)
        mixing_distribution.set_method(mixing_distribution_sampler)

        boom_components = list()
        mean_priors = list()
        sample_size_priors = list()
        for component in self._components:
            mean_prior = component["mean_prior"]
            sample_size_prior = component["sample_size_prior"]
            initial_a = mean_prior.mean * sample_size_prior.mean
            initial_b = sample_size_prior.mean - initial_a
            component_model = boom.BetaBinomialModel(initial_a, initial_b)
            if self._use_data_augmentation:
                component_sampler = boom.BetaBinomialPosteriorSampler(
                    component_model,
                    mean_prior.boom(),
                    sample_size_prior.boom())
                component_model.set_method(component_sampler)
            else:
                mean_priors.append(mean_prior.boom())
                sample_size_priors.append(sample_size_prior.boom())

            boom_components.append(component_model)

        boom_model = boom.BetaBinomialMixtureModel(
            boom_components, mixing_distribution)
        if self._use_data_augmentation:
            sampler = boom.BetaBinomialMixturePosteriorSampler(boom_model)
        else:
            sampler = boom.BetaBinomialMixtureDirectPosteriorSampler(
                boom_model, dirichlet_prior, mean_priors, sample_size_priors)
        boom_model.set_method(sampler)
        return boom_model

    def __getstate__(self):
        """
        Make the object pickle-able.
        """
        ans = self.__dict__.copy()
        ans["data"] = self._boom_model.data.to_numpy()
        ans["_boom_model"] = None
        return ans

    def __setstate__(self, state):
        """
        Restore the object from a pickle.
        """
        self.__dict__ = state.copy()
        self._boom_model = self._build_boom_model()
        data = np.array(state["data"]).copy()
        del self.__dict__["data"]
        self._boom_model.add_data(data)
        self.restore_params(-1)
