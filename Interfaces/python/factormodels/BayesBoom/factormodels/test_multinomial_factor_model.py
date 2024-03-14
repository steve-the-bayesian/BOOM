import unittest
import matplotlib.pyplot as plt


from BayesBoom.factormodels import (
    MultinomialFactorModel
)

# from BayesBoom.R import delete_if_present
import BayesBoom.R as R
import BayesBoom.test_utils as test_utils


import numpy as np
import pandas as pd
# import scipy.sparse
# import pickle
import string

# import matplotlib.pyplot as plt


def simulate_user_classes(num_users, probs):
    num_classes = probs.shape[0]
    user_values = np.random.choice(num_classes, num_users, p=probs)
    user_labels = random_strings(num_users, 10)
    return pd.Series(user_values, index=user_labels)


def simulate_site_params(num_sites, num_categories):
    values = np.random.rand(num_sites, num_categories)
    labels = random_strings(num_sites, 20)
    return pd.DataFrame(values, index=labels)


def random_strings(num_strings, length):
    """
    Return a numpy array of lowercase ASCII strings.
    Args:
      num_strings: The length of the returned array.
      length:  The length of each string in the array.
    """
    if num_strings > 26**length:
        raise Exception("Too many strings requested")
    letters = list(string.ascii_lowercase)
    letter_matrix = np.random.choice(letters, (num_strings, length))
    ans = np.array(["".join(x) for x in letter_matrix])
    while len(np.unique(ans)) < num_strings:
        levels, counts = np.unique(ans, return_counts=True)
        duplicates = levels[counts > 1]
        dup_counts = counts[counts > 1]
        for i in range(len(duplicates)):
            ans[ans == duplicates[i]] = random_strings(dup_counts[i], length)
    return ans


def simulate_pfm_data(user_classes, site_params):
    """
    Args:

      user_classes: A num_users-vector of values in 0, ..., K-1 indicating the
        class to which each user belongs.
      site_params: A num_sites by num_classes matrix giving the Poisson rate
        parameters for users in each category.
    """
    frames = []
    for i, level in enumerate(user_classes):
        lam = site_params.iloc[:, level]
        counts = np.random.poisson(lam)
        nvisits = np.sum(counts > 0)

        # Each frame is the data from a single user.
        frame = pd.DataFrame(
            {
                "user": np.full(nvisits, user_classes.index[i]),
                "site": site_params.index[counts > 0],
                "count": counts[counts > 0]
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
        self._site_params = simulate_site_params(num_sites, num_classes)
        self._data = simulate_pfm_data(self._user_classes, self._site_params)

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
        model.run_mcmc(niter=niter)
        print("Done with model run!")

    #     # =====================================================================
    #     # Check that the model is not updating the classes of the users marked
    #     # as known.
    #     known_user_draws = model.user_draws(known_users.index[0])
    #     self.assertEqual(known_user_draws.shape[0], niter)
    #     self.assertEqual(len(known_user_draws.shape), 1)
    #     known_user_true_value = known_users.iloc[0]
    #     self.assertTrue(np.alltrue(known_user_draws == known_user_true_value))

    #     user_idx = [3, 8, 12]
    #     some_users = known_users.index[user_idx]
    #     self.assertEqual(model.user_draws(some_users).shape, (niter, 3))

    #     # =====================================================================
    #     # For users that are unknown, check that the model is choosing the true
    #     # class as 'most likely' most of the time.
    #     ud = model.user_distribution()
    #     ud["truth"] = self._user_classes[model.user_ids]
    #     ud["chosen"] = np.argmax(ud.values[:, :4], axis=1)
    #     xtab = pd.crosstab(ud["truth"], ud["chosen"])

    #     self.assertGreater(np.sum(np.diag(xtab)), .9 * self.num_users)

    #     # =====================================================================
    #     # Check that the marginal distributions of the site parameters cover the
    #     # true values reasonably frequently.

    #     # Check for a single site.
    #     # sid = model._site_ids[-1]
    #     # lam = model.site_draws(sid)
    #     # truth = self._site_params.loc[sid, :]
    #     # R.BoxplotTrue(lam, truth=truth)
    #     # plt.show()

    #     # for m in models:
    #     #     self.plot_sites(m, 4, 5, m.description, style="box")
    #     #     self.plot_sites(m, 4, 5, m.description, style="ts")
    #     # plt.show()

    #     # Check for all sites.
    #     for a_model in models:
    #         num_lam = self.num_classes * self.num_sites
    #         all_lam = np.reshape(a_model._site_draws, (niter, num_lam))
    #         true_lam = np.reshape(self._site_params.loc[
    #             a_model.site_ids, :].values, (1, num_lam)).ravel()
    #         self.assertTrue(test_utils.check_mcmc_matrix(all_lam, true_lam))


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
    rig.test_mcmc()

    print("Goodbye, cruel world!")

else:
    if __name__ == "__main__":
        unittest.main()