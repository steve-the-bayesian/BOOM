import numpy as np
import pandas as pd
import BayesBoom.boom as boom
import BayesBoom.R as R
# import matplotlib.pyplot as plt


class Visitor:
    def __init__(self, boom_vistor_object):
        self._boom_visitor = boom_vistor_object

    @property
    def visits(self):
        return pd.Series(self._boom_visitor.visits)

    @property
    def number_of_distinct_sites_visited(self):
        return self._boom_visitor.number_of_distinct_sites_visited

    @property
    def imputed_class(self):
        return self._boom_visitor.imputed_class

    def __str__(self):
        ans = (
            f"User {self._boom_visitor.id} has visited "
            f"{self._boom_visitor.number_of_distinct_sites_visited} "
            f"distinct sites for a total of {self._boom_visitor.num_visits} "
            "visits."
        )
        return ans


class PoissonFactorModel:

    def __init__(self, nlevels, hierarchical_prior: bool = True):
        """
        Args:
          nlevels:  The number of potential values in the latent factor.

          hierarchical_prior: If True then use a hierarchical model for the
            prior distribution over the site parameters.  Otherwise assign each
            site parameter an independent Gamma prior.
        """

        self._nlevels = int(nlevels)
        self._model = boom.PoissonFactorModel(nlevels)
        self._site_ids = None
        self._user_ids = None

        if hierarchical_prior:
            self._hyperprior_parameters = {
                "Sigma_guess": np.eye(nlevels - 1),
                "prior_Sigma_sample_size": nlevels,
                "prior_mean": np.zeros(nlevels - 1),
                "prior_mean_sample_size": 1,
            }
        else:
            # _default_site_prior is a list of GammaModel objects giving the
            # prior distribution for each Poisson rate parameter in an arbitrary
            # site.  There is one GammaModel object for each level of the latent
            # factor.
            self._default_site_prior = [R.GammaModel(1.0, 1.0)] * nlevels

            # _site_specific_priors is a list of GammaModel objects giving the
            # prior distribution of a specific site.  Keyed by site_id, the
            # values are lists of priors -- one prior for each level of the
            # latent factor.
            self._site_specific_priors = {}

        self._prior_class_membership_probabilites = (
            np.full(nlevels, 1.0 / nlevels))

    @property
    def niter(self):
        """
        The number of MCMC iterations in the model.
        """
        if hasattr(self, "_site_draws"):
            return self._site_draws.shape[0]
        else:
            return 0

    @property
    def nlevels(self):
        """
        The number of potential values in the latent category.
        """
        return self._nlevels

    @property
    def num_categories(self):
        """
        The number of potential values in the latent category.  A synonym for
        nlevels.
        """
        return self._nlevels

    @property
    def num_users(self):
        """
        The number of users in the data set.
        """
        return self._model.num_users

    @property
    def num_sites(self):
        """
        The number of sites observed in the data set.
        """
        return self._model.num_sites

    @property
    def has_hierarchical_prior(self):
        """
        Returns True iff the model is using a hierarchical prior.
        """
        return hasattr(self, "_hyperprior_parameters")

    @property
    def user_ids(self):
        """
        The ID's of the stored users, in the order kept by the underlying C++
        object.  This is the order they're stored in self._user_classes.
        """
        if self._user_ids is None:
            self._user_ids = np.array(self._model.visitor_ids)
        return self._user_ids

    @property
    def site_ids(self):
        """
        The ID's of the sites, in the order they are kept by the underlying C++
        object.  This is the order used to store self._site_params.
        """
        if self._site_ids is None:
            self._site_ids = self._model.site_ids
        return self._site_ids

    @property
    def exposures(self):
        """
        The posterior distribution of the number of users from each category.
        Expressed as a matrix with rows corresponding to Monte Carlo draws.
        """
        if not hasattr(self, "_exposures"):
            self._exposures = np.empty((self.niter, self.nlevels))
            for k in range(self.nlevels):
                is_level = self._user_draws.astype(int) == k
                self._exposures[:, k] = np.sum(is_level, axis=1)
        return self._exposures

    @property
    def sum_lambda(self):
        """
        The posterior distribution over the sum of the intensity parameters,
        across all sites, for each category.
        """
        if not hasattr(self, "_sum_lambda"):
            # self._site_draws have shape [niter, num_sites, nlevels]
            self._sum_lambda = self._site_draws.sum(axis=1)

        return self._sum_lambda

    def add_data(self, user, site, count):
        """
        Args:
          user: a vector of strings giving the user ID.
          site:  a string containing a site ID.
          count:  The number of times the user visited that site.

        The three arguments must be vectors of the same length.
        """

        if len(user) != len(site):
            raise Exception(
                f"The 'user' ({len(user)}) and 'site' ({len(site)}) arguments "
                "must have the same length")
        if len(user) != len(count):
            raise Exception(
                f"The 'user' ({len(user)}) and 'site' ({len(count)}) arguments "
                "must have the same length")

        self._model.add_data(R.to_numpy(user).astype(str),
                             R.to_numpy(site).astype(str),
                             R.to_numpy(count).astype(int))

    def site(self, site_id: str):
        return self._model.site(site_id)

    def user(self, user_id: str):
        """
        Returns the BOOM model object for the requested user, or None if the
        requested user is not found.
        """
        return Visitor(self._model.user(user_id))

    def set_known_user_demographics(self, users: pd.Series):
        """
        Args:
          users: A pd.Series indexed by user ids containing their associated
            demographic categories as values.  The values are integers in the
            range 0 .. K-1, where K is the number of categories.

        Effects:
          The 'users' varaible is saved as self._known_users.
        """
        self._known_users = users

    def set_site_parameters(self, intensities):
        """
        Set the intensity parameters for one or more sites.

        Args:
          intensities: A dict, keyed by site id, containing vectors of intensity
            parameters.
        """
        values = np.concatenate([np.array(x).reshape(1, -1)
                                 for x in intensities.values()],
                                axis=0)
        self._model.set_site_parameters(
            list(intensities.keys()),
            R.to_boom_matrix(values))

    def set_site_priors(self, site_ids, prior_a, prior_b):
        """
        Set the prior distribution over Poisson model parameters for specific
        sites.

        Args:
          site_ids: A list-like sequence of strings identifying the sites to be
            set.
          prior_a: A matrix of positive numbers.  Rows correspond to sites and
            columns to levels of the latent factor.  Entries are the "shape"
            parameter in a Gamma(shape, scale) prior distribution.
          prior_b: A matrix of positive numbers.  Rows correspond to sites and
            columns to levels of the latent factor.  Entries are the "scale"
            parameter in a Gamma(shape, scale) prior distribution.
        """
        if self.has_hierarchical_prior:
            raise Exception("You cannot set site specific priors with a "
                            "hierarchical prior.")

        self._site_specific_priors = {
            "prior_a": pd.DataFrame(prior_a, index=site_ids),
            "prior_b": pd.DataFrame(prior_b, index=site_ids),
        }

    def set_hyperprior_parameters(
            self,
            Sigma_guess,
            prior_Sigma_sample_size,
            prior_mean,
            prior_mean_sample_size):
        """
        Sets the parameters of the hyperprior.  The hyperprior describes the
        distribution (across sites) of the multinomial logit transform of the
        intensity paramaeters (i.e. log (lambda[k] / lambda[0])) with category 0
        as the reference class.  This prior is a multivariate Normal
        distribution(mu, Sigma) with dimension equal to one less than the number
        of classes.

        The mathematical form of the prior on mu, Sigma is a "normal inverse
        Wishart distribution":

          Sigma ~ W^{-1}(Sigma_guess, prior_Sigma_sample_size)
          mu | Sigma ~ N(prior_mean | Sigma / prior_mean_sample_size)

        Args:
          Sigma_guess: A prior guess at Sigma.
          prior_mean_sample_size: The number of observations (e.g. number of
            sites) worth of weight to assign to Sigma_guess.  This number should
            be >= the number of categories.
          prior_mean: A prior guess at mu.  A vector of all 0's places no prior
            favor on any category.
          prior_mean_sample_size: The number of observations worth of weight to
            assign to prior_mean.
        """
        self._hyperprior_parameters = {
            "Sigma_guess": Sigma_guess,
            "prior_Sigma_sample_size": prior_Sigma_sample_size,
            "prior_mean": prior_mean,
            "prior_mean_sample_size": prior_mean_sample_size
        }

        if hasattr(self, "_site_specific_priors"):
            del self._site_specific_priors

        if hasattr(self, "_default_site_prior"):
            del self._default_site_prior

    def set_MH_threshold(self, threshold):
        """
        When the hierarchical prior is used, sites with at least 'threshold'
        observations in each category are updated using Metropolis Hastings
        sampling.  Sites that to not meet the threshold are updated using slice
        sampling.

        Args:
          threshold: The minimum number of observations needed to use MH
            updating.
        """
        if not self.has_hierarchical_prior:
            raise Exception("Switch to a hierarchical prior by calling "
                            "set_hyperprior_parameters.")
        self._hyperprior_parameters["MH_threshold"] = threshold

    def set_default_user_prior(self,
                               prior_weights: np.ndarray):
        """
        Args:
          prior_weights: A numpy array containing the discrete probability
            distribution over the latent categories.  This is the prior
            distribution to be used for each user's latent category, unless a
            different prior is explicitly set for that user.
        """
        if prior_weights.shape[0] != self._nlevels:
            raise Exception(
                f"prior_weights should have length {self._nlevels}.")
        self._prior_class_membership_probabilites = prior_weights

    def set_default_site_prior(self, prior):
        """
        Args:
          prior:  One of the following:
          - A single R.GammaModel object that will be used for all levels of all
            sites that do not have site-specific priors.
          - An iteratble of R.GammaModel objects with length equal to the number
            of levels of the latent categorical variable.
          - A 2 column numpy array, with rows corresponding to the levels of the
            latent categorical variable.  The first column contains the 'a'
            parameter of the priors.  The second contains the 'b' column.

        """
        if self.has_hierarchical_prior:
            del self._hyperprior_parameters
            # raise Exception("You cannot set a default site prior when using "
            #                 "a hierarchical site prior.")

        if isinstance(prior, R.GammaModel):
            prior = [prior] * self._nlevels
        elif isinstance(prior, np.ndarray):
            prior_list = [R.GammaModel(prior[i, 0], prior[i, 1])
                          for i in range(prior.shape[0])]
            prior = prior_list

        if not hasattr(prior, "__len__"):
            raise Exception("'prior' should be a list of R.GammaModel objects")
        if len(prior) != self._nlevels:
            raise Exception("'prior' should contain one prior for each level.")
        self._default_site_prior = prior

    def run_mcmc(self, niter, ping: int = -8675309):
        """
        Run a Markov chain Monte Carlo posterior sampling algorithm.

        Args:
          niter:  The number of iterations to run the sampler.
          ping: Print a status update every 'ping' iterations.  If ping <= 0 or
            if ping is None then no status updates are printed.
        """
        self._posterior_sampler = self._assign_sampler(self._model)
        self._allocate_space(niter)
        self._site_ids = self._model.site_ids
        if ping == -8675309:
            ping = max(1, int(niter / 10))
        for i in range(niter):
            R.print_timestamp(i, ping=ping)
            self._model.sample_posterior()
            self._record_draw(i)

    def prior_class_probabilities(self, user_id):
        """
        Return the discrete probability distribution describing the prior belief
        about the requested site.
        """
        return R.to_numpy(
            self._posterior_sampler.prior_class_probabilities(user_id))

    def posterior_class_probabilities(self,
                                      user_id,
                                      distribution=False,
                                      burn=0):
        """
        The posterior discrete probability distribution of a user's
        latent class.

        Args:
          user_id: Either a string specifying the id of a user, or an integer in
            the range [0, num_users).
          distribution: See below.
          burn: The number of MCMC iterations to discard as burn-in.

        Returns:
          If distribution is True then return a 2D numpy array containing the
          conditional posterior distribution of the class probabilities given
          the model parameters at each MCMC iteration.  Each row of this array
          is one MCMC iteration.  If False then a 1D numpy array is returned
          averaging over the posterior distribution of the parameters.
        """
        if isinstance(user_id, int):
            user_id = self._user_ids[user_id]

        prior = self.prior_class_probabilities(user_id)
        if prior.max() >= .9999:
            pass
        else:
            log_prior = np.log(prior)

        nlevels = len(prior)
        log_prior = np.log(prior)
        user_object = self.user(user_id)

        visits = user_object.visits
        nsites = len(visits)

        niter = self.niter - burn
        log_likelihood = np.zeros((niter, nlevels))

        for i in range(nsites):
            url = visits.index[i]
            num_visits = visits.values[i]
            lambda_draws = self.site_draws(url)[burn:, :]
            log_likelihood += num_visits * np.log(lambda_draws)

        log_likelihood -= self.sum_lambda[burn:, :]
        unnormalized_posterior = log_likelihood + log_prior.reshape((1, -1))

        iter_max = np.max(unnormalized_posterior, axis=1).reshape((-1, 1))
        unnormalized_posterior = np.exp(unnormalized_posterior - iter_max)
        normalizing_constant = np.sum(unnormalized_posterior,
                                      axis=1).reshape((-1, 1))
        posterior = unnormalized_posterior / normalizing_constant
        if distribution:
            return posterior
        else:
            return np.mean(posterior, axis=0)

    def user_draws(self, user_id):
        """
        Args:
          user_id: either a single user id (an int or a string), or a collection
            of id's (a list, numpy array, pandas series, or similar)

        Return:
          If user_id is an int or a string then the return value is a 1-d numpy
          array of Monte Carlo draws for that user.  If a collection then
          user_ids is a 2-d numpy array with rows representing Monte Carlo draws
          and columns aligning with the values in user_id.  That is, the first
          user is the first column, the second user is the second column, etc.
        """
        singleton = False
        if isinstance(user_id, str):
            user_id = [user_id]
            singleton = True

        idx = np.array(boom.fast_find(user_id, self.user_ids))
        if np.any(idx < 0):
            notfound = idx < 0
            num_notfound = np.sum(notfound)
            if num_notfound < 20:
                msg = "The following ID's were not found:\n"
                msg += f"{user_id[notfound]}"
            else:
                msg = f"{num_notfound} user ID's were not found."
            raise ValueError(msg)

        ans = self._user_draws[:, idx]
        if singleton:
            ans = ans.ravel()
        return ans

    def user_distribution(self, burn=None):
        """
        Returns a matrix [num_users x num_categories] giving the Monte Carlo
        estimate of the posterior probability that each user is in each
        category.
        """
        levels = np.arange(self.num_categories, dtype="float")
        user_counts = [R.table(self._user_draws[:, x]).reindex(
            levels, fill_value=0)
                       for x in range(self._user_draws.shape[1])]
        user_counts = pd.DataFrame(user_counts, index=self._user_ids)
        totals = user_counts.sum(axis=1)
        return user_counts.div(totals, axis=0)

    def site_draws(self, site_id):
        """
        The posterior distribution of the intensity parameters for a given
        site.  A matrix with rows corresponding to Monte Carlo draws, and
        columns to intensity parameter values for different latent levels.
        """
        idx = np.searchsorted(self._site_ids, site_id)
        if self._site_ids[idx] != site_id:
            raise ValueError(f"Site {site_id} could not be found.")
        return self._site_draws[:, idx, :]

    def impute_visitors(self):
        """
        Fill each user's latent category value with a draw from its posterior
        distribution given the current values of site intensity paramaeters.
        """
        self._posterior_sampler.impute_visitors()

    def draw_site_parameters(self):
        self._posterior_sampler.draw_site_parameters()

    def _assign_sampler(self, model):
        """
        Create, configure, and assign to the model a posterior sampler
        consistent with the previously supplied model options.

        Returns:
          The created posterior sampler.
        """

        if self.has_hierarchical_prior:
            sampler = self._create_hierarchical_sampler(model)
        else:
            sampler = self._create_independence_sampler(model)

        known_users = getattr(self, "_known_users", None)
        if known_users is not None:
            num_known = len(known_users)
            probs = np.zeros((num_known, self.nlevels))
            x = np.arange(num_known)
            probs[x, known_users.values] = 1.0
            sampler.set_prior_class_probabilities(
                known_users.index,
                probs)

        model.set_method(sampler)
        return sampler

    def _create_hierarchical_sampler(self, model):
        """
        Create and return a hierarchical sampler.
        """
        sampler = boom.PoissonFactorHierarchicalSampler(
            model,
            R.to_boom_vector(self._prior_class_membership_probabilites),
            prior_mean=R.to_boom_vector(
                self._hyperprior_parameters["prior_mean"]),
            kappa=self._hyperprior_parameters["prior_mean_sample_size"],
            Sigma_guess=R.to_boom_spd(
                self._hyperprior_parameters["Sigma_guess"]),
            prior_df=self._hyperprior_parameters["prior_Sigma_sample_size"])

        MH_threshold = self._hyperprior_parameters.get("MH_threshold", None)
        if MH_threshold:
            sampler.set_MH_threshold(MH_threshold)

        return sampler

    def _create_independence_sampler(self, model):
        """
        Create and return an independence sampler.
        """
        prior_a = np.array([x.a for x in self._default_site_prior])
        prior_b = np.array([x.b for x in self._default_site_prior])
        sampler = boom.PoissonFactorModelIndependentGammaPosteriorSampler(
            model,
            R.to_boom_vector(self._prior_class_membership_probabilites),
            R.to_boom_vector(prior_a),
            R.to_boom_vector(prior_b),
            boom.GlobalRng.rng)

        site_ids = self.site_ids
        prior_a = pd.DataFrame(np.empty((self.num_sites, self.num_categories)),
                               index=site_ids)
        prior_b = pd.DataFrame(np.empty((self.num_sites, self.num_categories)),
                               index=site_ids)
        for k, prior in enumerate(self._default_site_prior):
            prior_a.iloc[:, k] = prior.a
            prior_b.iloc[:, k] = prior.b

        if getattr(self, "_site_specific_priors", None):
            site_specific_ids = self._site_specific_priors["prior_a"].index
            prior_a.loc[site_specific_ids, :] = (
                self._site_specific_priors["prior_a"]
            )
            prior_b.loc[site_specific_ids, :] = (
                self._site_specific_priors["prior_b"]
            )

        sampler.set_site_priors(
            site_ids,
            R.to_boom_matrix(prior_a),
            R.to_boom_matrix(prior_b))

        return sampler

    def _allocate_space(self, niter):
        """
        Allocate space to hold 'niter' MCMC draws.
        """

        # users[i, j] is the imputed category for user j in iteration i.
        self._user_draws = np.empty((niter, self.num_users))

        # sites[i, j, k] is the poisson rate parameter for category k
        # on site j.
        self._site_draws = np.empty((niter, self.num_sites, self.nlevels))

        if self.has_hierarchical_prior:
            dim = self.nlevels - 1
            self._prior_mean_draws = np.empty((niter, dim))
            self._prior_variance_draws = np.empty((niter, dim, dim))

    def _record_draw(self, iteration: int):
        """
        Record the most recent MCMC draw.

        Args:
          iteration: The iteration number of the draw to be recorded.
        """
        self._user_draws[iteration, :] = np.array(self._model.imputed_classes)
        self._site_draws[iteration, :, :] = self._model.site_params.to_numpy()

        if hasattr(self, "_sum_lambda"):
            del self._sum_lambda

        if hasattr(self, "_exposures"):
            del self._exposures

        if self.has_hierarchical_prior:
            self._prior_mean_draws[iteration, :] = (
                self._posterior_sampler.hyperprior.mean.to_numpy()
            )
            self._prior_variance_draws[iteration, :, :] = (
                self._posterior_sampler.hyperprior.Sigma.to_numpy()
            )
