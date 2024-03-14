import numpy as np
import BayesBoom.boom as boom
import BayesBoom.R as R
from .factor_model_base import FactorModelBase


class MultinomialFactorModel(FactorModelBase):

    def __init__(self, nlevels):
        super().__init__(nlevels)
        self._model = boom.MultinomialFactorModel(int(nlevels))

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
