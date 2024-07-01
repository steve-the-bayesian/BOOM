import unittest
# import matplotlib.pyplot as plt
from BayesBoom.test_utils import random_strings
import BayesBoom.R as R

from BayesBoom.factormodels import (
    MultinomialFactorModel,
    MultinomialFactorModelJsonEncoder,
    MultinomialFactorModelJsonDecoder
)

import json
import pickle

# from BayesBoom.R import delete_if_present
# import BayesBoom.R as R
import BayesBoom.test_utils as test_utils


import numpy as np
import pandas as pd


def simulate_user_classes(num_users, probs):
    num_classes = probs.shape[0]
    user_values = np.random.choice(num_classes, num_users, p=probs)
    user_labels = random_strings(num_users, 10, ensure_unique=True)
    return pd.Series(user_values, index=user_labels)


def simulate_site_params(num_sites, num_categories, default_site_name="Other"):
    values = np.random.rand(num_sites, num_categories)
    labels = random_strings(num_sites, 20, ensure_unique=True)
    labels[0] = default_site_name
    totals = np.sum(values, axis=0)
    values = values / totals
    return pd.DataFrame(values, index=labels)


def simulate_multinomial_factor_data(user_classes, site_params):
    """
    Args:

      user_classes: A pd.Series of values in 0, ..., K-1 indicating the class
        to which each user belongs.  The index of the series is the user-id.
      site_params: A pd.DataFrame with num_sites rows and num_classes columns,
        giving the Poisson rate parameters for users in each category.  The
        index is the site-id.
    """
    frames = []
    for i, level in enumerate(user_classes):
        num_sites = 1 + np.random.poisson(5)
        sites_visited = np.random.choice(
            site_params.index,
            size=num_sites,
            replace=False,
            p=site_params.iloc[:, user_classes.iloc[i]])

        counts = np.full(num_sites, 1)

        # Each frame is the data from a single user.
        frame = pd.DataFrame(
            {
                "user": np.full(num_sites, user_classes.index[i]),
                "site": sites_visited,
                "count": counts
            }
        )
        frames.append(frame)

    return pd.concat(frames, axis=0, ignore_index=True)


class MultinomialFactorModelTest(unittest.TestCase):
    def setUp(self):
        np.random.seed(8675309)
        self.simulate_data(num_users=1000, num_sites=50, num_classes=4)

    def simulate_data(self, num_users, num_sites, num_classes):
        self._class_probs = np.random.randn(num_classes) ** 2
        self._class_probs = self._class_probs / np.sum(self._class_probs)
        self._user_classes = simulate_user_classes(
            num_users, self._class_probs)
        self.assertEqual(len(self._user_classes), num_users)
        self._site_params = simulate_site_params(num_sites, num_classes)
        self._data = simulate_multinomial_factor_data(
            self._user_classes, self._site_params)

    @property
    def num_classes(self):
        return self._site_params.shape[1]

    @property
    def num_users(self):
        return len(self._user_classes)

    @property
    def class_probs(self):
        return self._class_probs

    @property
    def num_sites(self):
        return self._site_params.shape[0]

    def smoke_test(self):
        model = MultinomialFactorModel(self.num_classes)
        self.assertIsInstance(model, MultinomialFactorModel)

    def build_model(self):
        model = MultinomialFactorModel(self.num_classes)
        model.add_data(user=self._data["user"],
                       site=self._data["site"],
                       count=self._data["count"])
        return model

    def test_prior_class_probabilities(self):
        model = self.build_model()
        users = np.unique(self._data["user"])
        probs = model.prior_class_probabilities(users[0])
        self.assertIsInstance(probs, np.ndarray)
        self.assertTrue(hasattr(probs, "shape"))
        self.assertEqual(len(probs.shape), 1)
        self.assertEqual(probs.shape[0], self.num_classes)

        probs = model.prior_class_probabilities(users)
        self.assertIsInstance(probs, np.ndarray)
        self.assertTrue(hasattr(probs, "shape"))
        self.assertEqual(len(probs.shape), 2)
        self.assertEqual(probs.shape,
                         (self.num_users, self.num_classes))

    def test_mcmc(self):
        model = self.build_model()

        # =====================================================================
        # Test that the data stored inside the C++ model agree with the data
        # stored in python.
        spot_check_sites = np.random.choice(self._site_params.index, 10)
        for sid in spot_check_sites:
            good = self._data["site"] == sid
            raw_count = self._data[good]["count"].sum()
            model_count = model._model.site(sid).num_visits
            self.assertEqual(raw_count, model_count)

        spot_check_users = np.random.choice(self._user_classes.index, 10)
        for uid in spot_check_users:
            good = self._data["user"] == uid
            raw_count = self._data[good]["count"].sum()
            model_count = model._model.user(uid).num_visits
            self.assertEqual(raw_count, model_count)

        # =====================================================================
        # Run the MCMC.
        num_known = 800
        known_users = self._user_classes.iloc[:num_known]
        niter = 1000
        model.set_known_user_demographics(known_users)
        model.set_num_threads(15)
        model.run_mcmc(niter=niter)
        print("Done with model run!")

        # =====================================================================
        # Check that the model is not updating the classes of the users marked
        # as known.
        known_user_draws = model.user_draws(known_users.index[0])
        self.assertEqual(known_user_draws.shape[0], niter)
        self.assertEqual(len(known_user_draws.shape), 1)
        known_user_true_value = known_users.iloc[0]
        self.assertTrue(np.alltrue(known_user_draws == known_user_true_value))

        user_idx = [3, 8, 12]
        some_users = known_users.index[user_idx]
        self.assertEqual(model.user_draws(some_users).shape, (niter, 3))

        # =====================================================================
        # For users that are unknown, check that the model is choosing the true
        # class as 'most likely' most of the time.
        ud = model.user_distribution()
        ud["truth"] = self._user_classes[model.user_ids]
        ud["chosen"] = np.argmax(ud.values[:, :4], axis=1)
        xtab = pd.crosstab(ud["truth"], ud["chosen"])

        self.assertGreater(np.sum(np.diag(xtab)), .9 * self.num_users)

        # =====================================================================
        # Check that the marginal distributions of the site parameters cover the
        # true values reasonably frequently.

        # Check for a single site.
        # sid = model._site_ids[-1]
        # lam = model.site_draws(sid)
        # truth = self._site_params.loc[sid, :]
        # R.BoxplotTrue(lam, truth=truth)
        # plt.show()

        # for m in models:
        #     self.plot_sites(m, 4, 5, m.description, style="box")
        #     self.plot_sites(m, 4, 5, m.description, style="ts")
        # plt.show()

        # Check for all sites.
        num_probs = self.num_classes * self.num_sites
        all_params = np.reshape(model._site_draws, (niter, num_probs))
        true_params = np.reshape(self._site_params.loc[
            model.site_ids, :].values, (1, num_probs)).ravel()
        self.assertTrue(test_utils.check_mcmc_matrix(
            all_params, true_params))

    def test_predictions(self):
        model = self.build_model()

        # Run the MCMC.
        num_known = 800
        known_users = self._user_classes.iloc[:num_known]
        niter = 1000
        model.set_known_user_demographics(known_users)
        model.set_nthreads(14)
        model.run_mcmc(niter=niter)
        user_ids = model.user_ids[-5:-1]
        probs = model.posterior_class_probabilities(user_ids)
        data_subset = self._data[self._data["user"].isin(user_ids)]

        model.set_default_site_name(model.site_ids[0])

        priors = pd.DataFrame(
            model.prior_class_probabilities(user_ids),
            index=user_ids)

        probs2 = model.infer_posterior_distributions(
            data_subset["user"],
            data_subset["site"],
            priors=priors)

        print("probs = \n", probs)
        print("probs2 = \n", probs2)

        self.assertTrue(np.allclose(probs, probs2))
        print("Done with model run!")

    def test_json(self):
        model = self.build_model()
        num_known = 800
        known_users = self._user_classes.iloc[:num_known]
        niter = 100
        model.set_known_user_demographics(known_users)
        model.run_mcmc(niter=niter)

        json_string = json.dumps(model, cls=MultinomialFactorModelJsonEncoder)
        other_model = json.loads(json_string,
                                 cls=MultinomialFactorModelJsonDecoder)
        self.assertIsInstance(other_model, MultinomialFactorModel)
        # The deserialized model is functionally equivalent to the original
        # model if they produce the same posterior distributions.
        test_users = self._data["user"].unique()[:3]
        test_data = self._data[self._data["user"].isin(test_users)]

        post1 = model.infer_posterior_distributions(
            test_data["user"], test_data["site"])
        post2 = other_model.infer_posterior_distributions(
            test_data["user"], test_data["site"])
        self.assertTrue(np.allclose(post1, post2))

        # Now do the same check with data omitted.
        model.omit_data_when_serializing()
        json_string = json.dumps(model, cls=MultinomialFactorModelJsonEncoder)
        other_model = json.loads(json_string,
                                 cls=MultinomialFactorModelJsonDecoder)
        self.assertIsInstance(other_model, MultinomialFactorModel)
        post1 = model.infer_posterior_distributions(
            test_data["user"], test_data["site"])
        post2 = other_model.infer_posterior_distributions(
            test_data["user"], test_data["site"])
        self.assertTrue(np.allclose(post1, post2))

        # Check that we can do posterior inference with data that was not
        # part of the training set.
        num_test_users = 100
        test_user_names = random_strings(num_test_users, 20, ensure_unique=True)
        test_user_classes = pd.Series(
            np.random.choice(range(model.nlevels), 100),
            index=test_user_names)
        more_test_data = simulate_multinomial_factor_data(
            test_user_classes, self._site_params)
        post1 = model.infer_posterior_distributions(
            more_test_data["user"], more_test_data["site"])
        post2 = other_model.infer_posterior_distributions(
            more_test_data["user"], more_test_data["site"])
        self.assertTrue(np.allclose(post1, post2))

    def test_pickle(self):
        model = self.build_model()
        num_known = 800
        known_users = self._user_classes.iloc[:num_known]
        niter = 100
        model.set_known_user_demographics(known_users)
        model.run_mcmc(niter=niter)

        fname = "multinomial_factor_model.pkl"
        with open(fname, "wb") as pkl:
            pickle.dump(model, pkl)

        with open(fname, "rb") as pkl:
            other_model = pickle.load(pkl)
        R.delete_if_present(fname)
        self.assertIsInstance(other_model, MultinomialFactorModel)

        # The deserialized model is functionally equivalent to the original
        # model if they produce the same posterior distributions.
        test_users = self._data["user"].unique()[:3]
        test_data = self._data[self._data["user"].isin(test_users)]

        post1 = model.infer_posterior_distributions(
            test_data["user"], test_data["site"])
        post2 = other_model.infer_posterior_distributions(
            test_data["user"], test_data["site"])
        self.assertTrue(np.allclose(post1, post2))

        # Now run the same test code in "light mode" after omitting data.
        model.omit_data_when_serializing()
        fname = "multinomial_factor_model.pkl"
        with open(fname, "wb") as pkl:
            pickle.dump(model, pkl)
        with open(fname, "rb") as pkl:
            other_model = pickle.load(pkl)
        R.delete_if_present(fname)
        self.assertIsInstance(other_model, MultinomialFactorModel)

        post1 = model.infer_posterior_distributions(
            test_data["user"], test_data["site"])
        post2 = other_model.infer_posterior_distributions(
            test_data["user"], test_data["site"])
        self.assertTrue(np.allclose(post1, post2))

        # Check that we can do posterior inference with data that was not
        # part of the training set.
        num_test_users = 100
        test_user_names = random_strings(num_test_users, 20, ensure_unique=True)
        test_user_classes = pd.Series(
            np.random.choice(range(model.nlevels), 100),
            index=test_user_names)
        more_test_data = simulate_multinomial_factor_data(
            test_user_classes, self._site_params)
        post1 = model.infer_posterior_distributions(
            more_test_data["user"], more_test_data["site"])
        post2 = other_model.infer_posterior_distributions(
            more_test_data["user"], more_test_data["site"])
        self.assertTrue(np.allclose(post1, post2))

_debug_mode = True

if _debug_mode:
    import pdb  # noqa

    # Turn warnings into errors.
    # import warnings
    # warnings.simplefilter("error")

    # Run the test you are trying to debug here.  Instantiate the test class,
    # then call the problematic test.  Call pdb.pm() in the event of an
    # exception.
    print("Hello, world!")

    rig = MultinomialFactorModelTest()
    if hasattr(rig, "setUpClass"):
        rig.setUpClass()
    if hasattr(rig, "setUp"):
        rig.setUp()

    rig.smoke_test()
    rig.test_json()
    rig.test_pickle()

    print("Goodbye, cruel world!")

else:
    if __name__ == "__main__":
        unittest.main()
