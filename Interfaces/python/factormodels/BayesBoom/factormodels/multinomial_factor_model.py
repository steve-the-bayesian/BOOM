import numpy as np
import pandas as pd
import BayesBoom.boom as boom
import BayesBoom.R as R
from .factor_model_base import FactorModelBase
import json


class MultinomialFactorModel(FactorModelBase):

    def __init__(self, nlevels):
        super().__init__(nlevels)
        self._model = boom.MultinomialFactorModel(int(nlevels))
        self._default_site_name = "Other"

    @property
    def has_hierarchical_prior(self):
        return False

    def set_default_site_name(self, default_site_name):
        self._default_site_name = default_site_name

    @property
    def default_site_name(self):
        return self._default_site_name

    def site_draws(self, site_id):
        idx = np.searchsorted(self._site_ids, site_id)
        if self._site_ids[idx] != site_id:
            raise ValueError(f"Site {site_id} could not be found.")
        return self._site_draws[:, idx, :]

    def assign_classes(self, user_ids, burn=0):
        """
        Assign class indicators to a group of user id's.  The assignment
        balances each user's marginal posterior class probabilities against
        the need to match the global prior distribution.
        """
        posterior_probs = self.posterior_class_probabilities(
            user_id=user_ids, distribution=False, burn=burn)
        assignment = R.assign_classes(posterior_probs,
                                      self._prior_class_membership_probabilites)
        return pd.Series(assignment, index=user_ids)

    def infer_posterior_distributions(self,
                                      user_ids,
                                      sites_visited,
                                      priors=None,
                                      burn: int = 0):
        """
        Args:
          user_ids: A column of unique identifiers (strings) of the same length
            as 'sites_visited'.
          sites_visited: A vector of strings giving the names of the sites
            visited by each user.
          priors: A pd.DataFrame indexed by user_id.  If a user_id in user_ids
            is present in priors, then that row is used as the prior
            distribution when computing that user's posterior distribution of
            class membership.  If a user_id is not present (or if priors is
            None) then the default demographic prior from the model is used
            instead.
          burn:  The number of initial MCMC iterations to discard as burn-in.

        Returns:
          A pd.DataFrame, indexed by the unique values in user_ids, containing
          the posterior probabilities of class membership for that user in the
          columns.
        """

        unique_user_ids = np.unique(user_ids)
        default_prior = self._prior_class_membership_probabilites
        if priors is None:
            priors = pd.DataFrame(
                np.array([default_prior] * len(unique_user_ids)),
                index=unique_user_ids)

        if not isinstance(priors, pd.DataFrame):
            raise Exception("'priors' must be a pandas DataFrame.")

        # get default_site_name, default_prior
        if (burn < 0) or (not burn):
            burn = 0

        tmp_probs = self._model.infer_posterior_distributions(
                priors.index.tolist(),
                R.to_boom_matrix(priors.values),
                default_prior=R.to_boom_vector(
                    self._prior_class_membership_probabilites),
                user_ids=user_ids.tolist(),
                sites_visited=sites_visited.tolist(),
                default_site_name=str(self.default_site_name),
                site_draws=self._site_draws,
                burn=burn)

        probs = R.to_pd_dataframe(tmp_probs)

        return probs

    def posterior_class_probabilities(self,
                                      user_id,
                                      distribution=False,
                                      burn=0):
        """
        The posterior discrete probability distribution of a user's latent
        class. This is for users known to the model during model-fitting time.
        For previously unseen users please use 'infer_posterior_distributions'.

        Args:
          user_id: Either a string specifying the id of a user, or an integer in
            the range [0, num_users).  Alternatively, an interable of such
            identifiers can be provided.
          distribution: See below.
          burn: The number of MCMC iterations to discard as burn-in.

        Returns:
          If distribution is True then return a 2D numpy array containing the
          conditional posterior distribution of the class probabilities given
          the model parameters at each MCMC iteration.  Each row of this array
          is one MCMC iteration.  If False then a 1D numpy array is returned
          averaging over the posterior distribution of the parameters.

          If user_id is an iterable (other than a string) then the dimension of
          the returned array increases by 1.  In this case the leading dimension
          corresponds to the user id.
        """
        if isinstance(user_id, int):
            user_id = self._user_ids[user_id]
        elif isinstance(user_id, str):
            user_id = user_id

        if not R.is_iterable(user_id):
            user_id = [user_id]

        ans = self._model.posterior_class_probabilities(
            self._posterior_sampler,
            user_id,
            self._site_draws,
            int(burn)).to_numpy()

        return pd.DataFrame(ans, index=user_id)

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
                R.to_boom_matrix(probs))

        model.set_method(posterior_sampler)
        return posterior_sampler

    def __getstate__(self):
        payload = {
            "nlevels": self.nlevels,
            "user_ids": self._user_ids,
            "site_ids": self._site_ids,
            "user_draws": self._user_draws,
            "site_draws":  self._site_draws,
            "default_site_name": self._default_site_name,
            "known_users": self._known_users,
            "prior_class_probabilities":
            self._prior_class_membership_probabilites,
            "behavioral_data": self._model.extract_data()
        }
        return payload

    def __setstate__(self, payload):
        self._model = boom.MultinomialFactorModel(int(payload["nlevels"]))
        self._user_ids = payload["user_ids"]
        self._site_ids = payload["site_ids"]
        self._user_draws = np.array(payload["user_draws"])
        self._site_draws = np.array(payload["site_draws"])
        self._default_site_name = payload["default_site_name"]
        self._known_users = payload["known_users"]

        self._prior_class_membership_probabilites = payload[
            "prior_class_probabilities"]

        behavioral_data = payload.get("behavioral_data", None)
        if behavioral_data is not None:
            self.add_data(
                behavioral_data[0],
                behavioral_data[1],
                behavioral_data[2])


class MultinomialFactorModelJsonEncoder(json.JSONEncoder):
    """
    Overrides the 'default' method from JSONEncoder.  This allows a model object
    to be encoded using the standard algorithm:

    encoded_model = json.dumps(model, cls=MultinomialFactorModelJsonEncoder)
    """

    def default(self, obj):
        """
        Args:
          obj: A MultinomialFactorModel object to be encoded.

        Returns a JSON-encoded string containing the model artifacts.
        """

        series_encoder = R.PdSeriesJsonEncoder()

        # Everything in 'payload' must be a JSON-encodable object.
        payload = {
            "nlevels": obj.nlevels,
            "user_ids": obj._user_ids,
            "site_ids": obj._site_ids,
            "user_draws": obj._user_draws.tolist(),
            "site_draws":  obj._site_draws.tolist(),
            "default_site_name": obj._default_site_name,
            "known_users": series_encoder.default(obj._known_users),
            "prior_class_probabilities":
            obj._prior_class_membership_probabilites.tolist(),
            "behavioral_data": obj._model.extract_data()
        }

        return json.loads(json.dumps(payload))


class MultinomialFactorModelJsonDecoder(json.JSONDecoder):
    """
    Overrides the "decode" method from JSONDecoder.  This allows a model object
    to be deserialized from json_string using the standard algorithm:

    model = json.loads(json_string, cls=MultinomialFactorModelJsonDecoder)
    """

    def decode(self, json_string):
        payload = json.loads(json_string)
        return self.decode_from_dict(payload)

    def decode_from_dict(self, payload):
        model = MultinomialFactorModel(int(payload["nlevels"]))
        model._user_ids = payload["user_ids"]
        model._site_ids = payload["site_ids"]
        model._user_draws = np.array(payload["user_draws"])
        model._site_draws = np.array(payload["site_draws"])
        model._default_site_name = payload["default_site_name"]

        series_decoder = R.PdSeriesJsonDecoder()
        model._known_users = series_decoder.decode_from_dict(
            payload["known_users"])

        model._prior_class_membership_probabilites = np.array(
            payload["prior_class_probabilities"])

        behavioral_data = payload.get("behavioral_data", None)
        if behavioral_data is not None:
            model.add_data(
                behavioral_data[0],
                behavioral_data[1],
                behavioral_data[2])

        return model
