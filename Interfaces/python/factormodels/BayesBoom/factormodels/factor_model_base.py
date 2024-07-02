import numpy as np
import pandas as pd
import BayesBoom.boom as boom
import BayesBoom.R as R


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


class FactorModelBase:
    def __init__(self, nlevels):

        self._site_ids = None
        self._user_ids = None

        self._prior_class_membership_probabilites = (
            np.full(nlevels, 1.0 / nlevels))

        self._known_users = pd.Series()

    @property
    def model(self):
        if hasattr(self, "_model"):
            return self._model
        else:
            return None

    @property
    def nlevels(self):
        return self._prior_class_membership_probabilites.shape[0]

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
    def num_categories(self):
        """
        The number of potential values in the latent category.  A synonym for
        nlevels.
        """
        return self.nlevels

    @property
    def num_users(self):
        """
        The number of users in the data set.
        """
        if hasattr(self, "_model"):
            return self._model.num_users
        else:
            return 0

    @property
    def num_sites(self):
        """
        The number of sites observed in the data set.
        """
        if hasattr(self, "_model"):
            return self._model.num_sites
        else:
            return 0

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

    def add_data(self, user, site, count, max_chunk_size: int = 1000000):
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

        user = R.to_numpy(user).astype(str)
        site = R.to_numpy(site).astype(str)
        count = R.to_numpy(count).astype(int)

        print("Adding data to the boom model.")
        cursor = 0
        nrows = len(user)
        while cursor + max_chunk_size < nrows:
            print("Adding data chunk starting at row ", cursor, ".")
            end = cursor + max_chunk_size
            self._model.add_data(user[cursor:end].astype(str),
                                 site[cursor:end].astype(str),
                                 count[cursor:end].astype(int))
            cursor += max_chunk_size

        if cursor < nrows:
            print("Done with chunks.  Adding final piece with "
                  f"{nrows - cursor} rows.")
            self._model.add_data(user[cursor:nrows].astype(str),
                                 site[cursor:nrows].astype(str),
                                 count[cursor:nrows].astype(int))
        print("Done adding data!")

    def site(self, site_id: str):
        if self.model:
            return self.model.site(site_id)
        else:
            return None

    def user(self, user_id: str):
        """
        Returns the BOOM model object for the requested user, or None if the
        requested user is not found.
        """
        if hasattr(self, "_model"):
            return Visitor(self.model.user(user_id))
        else:
            return None

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

    def set_default_user_prior(self,
                               prior_weights: np.ndarray):
        """
        Args:
          prior_weights: A numpy array containing the discrete probability
            distribution over the latent categories.  This is the prior
            distribution to be used for each user's latent category, unless a
            different prior is explicitly set for that user.
        """
        if prior_weights.shape[0] != self.nlevels:
            raise Exception(
                f"prior_weights should have length {self.nlevels}.")
        self._prior_class_membership_probabilites = prior_weights

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

        Args:
          user_id: Either a string identifying the user, or a list of strings
            identifying a group of users.

        Returns:
          If user_id is a single string, then a 1-D numpy array of probabilities
          is returned.  If user_id is a list of strings then a 2D array is
          returned, with rows corresponding to user id and columns to class
          levels.
        """
        if hasattr(self, "_posterior_sampler"):
            return R.to_numpy(
                self._posterior_sampler.prior_class_probabilities(user_id))
        else:
            if isinstance(user_id, str):
                return self._prior_class_membership_probabilites
            else:
                return np.array(
                    [self._prior_class_membership_probabilites]
                    * len(user_id))

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
