import numpy as np
import pandas as pd
import BayesBoom.boom as boom
import BayesBoom.R as R
# import matplotlib.pyplot as plt


class PoissonFactorModel:

    def __init__(self, nlevels):
        """
        Args:
          nlevels:  The number of potential values in the latent factor.
        """

        self._nlevels = int(nlevels)
        self._model = boom.PoissonFactorModel(nlevels)
        self._site_ids = None
        self._user_ids = None

        # _default_site_prior is a list of GammaModel objects giving the prior
        # distribution for each Poisson rate parameter in an arbitrary site.
        # There is one GammaModel object for each level of the latent factor.
        self._default_site_prior = [R.GammaModel(1.0, 1.0)] * nlevels

        # _site_specific_priors is a list of GammaModel objects giving the prior
        # distribution of a specific site.  Keyed by site_id, the values are
        # lists of priors -- one prior for each level of the latent factor.
        self._site_specific_priors = {}

        self._prior_class_membership_probabilites = (
            np.full(nlevels, 1.0 / nlevels))

    @property
    def nlevels(self):
        return self._nlevels

    @property
    def num_categories(self):
        return self._nlevels

    @property
    def num_users(self):
        return self._model.num_users

    @property
    def num_sites(self):
        return self._model.num_sites

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
        ans = self._model.site(site_id)
        return ans

    def user(self, user_id: str):
        """
        Returns the BOOM model object for the requested user, or None if the
        requested user is not found.
        """
        ans = self._model.user(user_id)
        return ans

    def set_known_user_demographics(self, users: pd.Series):
        """
        Args:
          users: A list of users and their associated demographic categories
            (integers in the range 0 .. K-1, where K is the number of
            categories).

        Effects:
          The 'users' varaible is saved as self._known_users.
        """
        self._known_users = users

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
        self._site_specific_priors = {
            "prior_a": pd.DataFrame(prior_a, index=site_ids),
            "prior_b": pd.DataFrame(prior_b, index=site_ids),
        }

    def _initialize_model(self, nlevels: int):
        self._model = boom.PoissonFactorModel(self._nlevels)

    def set_default_user_prior(self,
                               prior_weights: np.ndarray):
        if prior_weights.shape[0] != self._nlevels:
            raise Exception(
                f"prior_weights should have length {self._nlevels}.")
        self._prior_class_membership_probabilites = prior_weights

    def set_default_site_prior(self, prior):
        """
        self._default_site_prior is a list of GammaModel objects giving the
        prior shape and scale parameters for the Poisson rate in each latent
        category.
        """

        if isinstance(prior, R.GammaModel):
            prior = [prior] * self._nlevels
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
        self._assign_sampler(self._model)
        self._allocate_space(niter)
        self._site_ids = self._model.site_ids
        if ping == -8675309:
            ping = max(1, int(niter / 10))
        for i in range(niter):
            R.print_timestamp(i, ping=ping)
            self._model.sample_posterior()
            self._record_draw(i)

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
        levels = np.arange(self.num_categories, dtype="float")
        user_counts = [R.table(self._user_draws[:, x]).reindex(
            levels, fill_value=0)
                       for x in range(self._user_draws.shape[1])]
        user_counts = pd.DataFrame(user_counts, index=self._user_ids)
        totals = user_counts.sum(axis=1)
        return user_counts.div(totals, axis=0)

    def site_draws(self, site_id):
        idx = np.searchsorted(self._site_ids, site_id)
        if self._site_ids[idx] != site_id:
            raise ValueError(f"Site {site_id} could not be found.")
        return self._site_draws[:, idx, :]

    def impute_visitors(self):
        self._posterior_sampler.impute_visitors()

    def draw_site_parameters(self):
        self._posterior_sampler.draw_site_parameters()

    def _assign_sampler(self, model):
        sampler = boom.PoissonFactorModelPosteriorSampler(
            model,
            R.to_boom_vector(
                self._prior_class_membership_probabilites),
            boom.GlobalRng.rng)
        self._model.set_method(sampler)
        self._posterior_sampler = sampler

        known_users = getattr(self, "_known_users", None)
        if known_users is not None:
            num_known = len(known_users)
            probs = np.zeros((num_known, self.nlevels))
            num_users = probs.shape[0]
            x = np.arange(num_users)
            probs[x, known_users.values] = 1.0
            sampler.set_prior_class_probabilities(
                known_users.index,
                probs)

        site_ids = self.site_ids
        prior_a = pd.DataFrame(np.empty((self.num_sites, self.num_categories)),
                               index=site_ids)
        prior_b = pd.DataFrame(np.empty((self.num_sites, self.num_categories)),
                               index=site_ids)
        for k, prior in enumerate(self._default_site_prior):
            prior_a.iloc[:, k] = prior.a
            prior_b.iloc[:, k] = prior.b

        if self._site_specific_priors:
            site_specific_ids = self._site_specific_priors["prior_a"].index
            prior_a.loc[site_specific_ids, :] = (
                self._site_specific_priors["prior_a"]
            )
            prior_b.loc[site_specific_ids, :] = (
                self._site_specific_priors["prior_b"]
            )

        model.set_site_priors(
            site_ids,
            R.to_boom_matrix(prior_a),
            R.to_boom_matrix(prior_b))

    def _allocate_space(self, niter):
        # users[i, j] is the imputed category for user j in iteration i.
        self._user_draws = np.empty((niter, self.num_users))

        # sites[i, j, k] is the poisson rate parameter for category k
        # on site j.
        self._site_draws = np.empty((niter, self.num_sites, self.nlevels))

    def _record_draw(self, iteration):
        self._user_draws[iteration, :] = np.array(self._model.imputed_classes)
        self._site_draws[iteration, :, :] = self._model.site_params.to_numpy()
