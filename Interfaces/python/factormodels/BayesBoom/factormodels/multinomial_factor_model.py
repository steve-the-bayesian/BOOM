import numpy as np
import BayesBoom.boom as boom
import BayesBoom.R as R
from .factor_model_base import FactorModelBase


class MultinomialFactorModel(FactorModelBase):

    def __init__(self, nlevels):
        super().__init__(nlevels)
        self._model = boom.MultinomialFactorModel(int(nlevels))

    @property
    def has_hierarchical_prior(self):
        return False

    def site_draws(self, site_id):
        idx = np.searchsorted(self._site_ids, site_id)
        if self._site_ids[idx] != site_id:
            raise ValueError(f"Site {site_id} could not be found.")
        return self._site_draws[:, idx, :]

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
            return prior
        else:
            log_prior = np.log(prior)

        nlevels = len(prior)
        user_object = self.user(user_id)

        visits = user_object.visits
        nsites = len(visits)

        niter = self.niter - burn
        log_likelihood = np.zeros((niter, nlevels))

        for i in range(nsites):
            url = visits.index[i]
            probs = self.site_draws(url)[burn:, :]
            log_likelihood += np.log(probs)

        unnormalized_posterior = log_likelihood + log_prior.reshape((1, -1))
        iter_max = np.max(unnormalized_posterior, axis=1).reshape((-1, 1))
        unnormalized_posterior = np.exp(
            unnormalized_posterior - np.exp(iter_max))
        normalizing_constant = np.sum(unnormalized_posterior, axis=1).reshape(
            (-1, 1))
        posterior = unnormalized_posterior / normalizing_constant
        if distribution:
            return posterior
        else:
            return np.mean(posterior, axis=0)

    def _allocate_space(self, niter: int):
        # users[i, j] is the imputed category for user j in iteration i.
        self._user_draws = np.empty((niter, self.num_users))

        # sites[i, j, k] is the poisson rate parameter for category k
        # on site j.
        self._site_draws = np.empty((niter, self.num_sites, self.nlevels))

    def _record_draw(self, iteration: int):
        self._user_draws[iteration, :] = np.array(self._model.imputed_classes)
        self._site_draws[iteration, :, :] = self._model.site_params.to_numpy()

    def _assign_sampler(self, model):
        posterior_sampler = boom.MultinomialFactorModelPosteriorSampler(
            model,
            R.to_boom_vector(self._prior_class_membership_probabilites))

        known_users = getattr(self, "_known_users", None)
        if known_users is not None and len(known_users) > 0:
            num_known = len(known_users)
            probs = np.zeros((num_known, self.nlevels))
            x = np.arange(num_known)
            probs[x, known_users.values] = 1.0
            posterior_sampler.set_prior_class_probabilities(
                known_users.index,
                probs)

        model.set_method(posterior_sampler)
        return posterior_sampler

