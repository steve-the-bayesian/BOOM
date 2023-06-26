import unittest

from BayesBoom.factormodels import (
    PoissonFactorModel
)

# from BayesBoom.R import delete_if_present
import BayesBoom.R as R

import numpy as np
import pandas as pd
# import scipy.sparse
# import pickle
import string

import matplotlib.pyplot as plt


def simulate_user_classes(num_users, probs):
    num_classes = probs.shape[0]
    return np.random.choice(num_classes, num_users, p=probs)


def simulate_site_params(num_sites, num_categories):
    ans = np.random.rand(num_sites, num_categories)
    return ans


def random_strings(num_strings, length):
    letters = list(string.ascii_lowercase)
    letter_matrix = np.random.choice(letters, (num_strings, length))
    return np.array(["".join(x) for x in letter_matrix])


def simulate_pfm_data(user_classes, site_params):
    """
    Args:

      user_classes: A num_users-vector of values in 0, ..., K-1 indicating the
        class to which each user belongs.
      site_params: A num_sites by num_classes matrix giving the Poisson rate
        parameters for users in each category.
    """
    num_users = len(user_classes)
    num_sites = site_params.shape[0]

    user_ids = random_strings(num_users, 10)
    site_ids = random_strings(num_sites, 20)

    frames = []

    for i, z in enumerate(user_classes):
        lam = site_params[:, z]
        counts = np.random.poisson(lam)
        nvisits = np.sum(counts > 0)

        frame = pd.DataFrame(
            {
                "user": np.full(nvisits, user_ids[i]),
                "site": site_ids[counts > 0],
                "count": counts[counts > 0]
            }
        )
        frames.append(frame)

    observed = pd.concat(frames, axis=0, ignore_index=True)
    return (
        observed,
        pd.Series(user_classes, index=user_ids),
        pd.DataFrame(site_params, index=site_ids)
    )


class PoissonFactorModelTest(unittest.TestCase):
    def setUp(self):
        np.random.seed(8675309)

        num_users = 100
        num_sites = 50
        num_classes = 4
        class_probs = np.random.randn(num_classes) ** 2
        class_probs = class_probs / np.sum(class_probs)
        self._class_probs = class_probs
        self._user_classes = simulate_user_classes(
            num_users, self._class_probs)
        self._site_params = simulate_site_params(num_sites, num_classes)

        (
            self._data,
            self._user_classes,
            self._site_params
        ) = simulate_pfm_data(self._user_classes, self._site_params)

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

    def test_mcmc(self):
        model = PoissonFactorModel(self.num_classes)
        model.set_default_site_prior(
            R.GammaModel(a=1, b=1))
        model.set_default_user_prior(
            np.full(self.num_classes, 1.0 / self.num_classes))
        model.add_data(user=self._data["user"],
                       site=self._data["site"],
                       count=self._data["count"])

        num_known = 50

        known_users = self._user_classes.iloc[:num_known]
        model.set_known_user_demographics(known_users)
        niter = 1000
        model.run_mcmc(niter=niter)
        # Check that the model is not updating the classes of the users marked
        # as known.
        known_user_draws = model.user(known_users.index[0])
        known_user_true_value = known_users.iloc[0]
        self.assertTrue(np.alltrue(known_user_draws == known_user_true_value))

        # For users that are unknown, check that the model is choosing the true
        # class as 'most likely' most of the time.
        success_count = 0
        trial_count = 0
        for i in range(num_known, len(self._user_classes)):
            trial_count += 1
            uid = self._user_classes.index[i]
            unknown_user_draws = model.user(uid)
            unknown_user_true_value = self._user_classes[uid]
            count_distribution = R.table(unknown_user_draws)
            success_count += (
                count_distribution.argmax() == unknown_user_true_value
            )

        self.assertGreater(success_count / trial_count, .5)

        sid = model._site_ids[-1]
        lam = model.site(sid)
        truth = self._site_params.loc[sid, :]
        R.BoxplotTrue(lam, truth=truth)

        num_lam = self.num_classes * self.num_sites
        all_lam = np.reshape(model._sites, (niter, num_lam))
        true_lam = np.reshape(self._site_params[model.site_ids].values, (1, num_lam)).ravel()
        bounds = R.qantile(all_lam, (.025, .975))
        above = all_lam > bounds.iloc[0, :]
        below = all_lam < bounds.iloc[1, :]
        inside = above & below

        import pdb
        pdb.set_trace()
        print("here")


_debug_mode = True

if _debug_mode:
    import pdb  # noqa

    # Turn warnings into errors.
    import warnings
    warnings.simplefilter("error")

    # Run the test you are trying to debug here.  Instantiate the test class,
    # then call the problematic test.  Call pdb.pm() in the event of an
    # exception.
    print("Hello, world!")

    rig = PoissonFactorModelTest()
    if hasattr(rig, "setUpClass"):
        rig.setUpClass()
    if hasattr(rig, "setUp"):
        rig.setUp()

    rig.test_mcmc()

    print("Goodbye, cruel world!")

if __name__ == "__main__":
    unittest.main()
