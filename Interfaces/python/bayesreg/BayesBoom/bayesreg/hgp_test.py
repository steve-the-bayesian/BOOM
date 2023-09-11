import unittest
import numpy as np
import pandas as pd

# pylint: disable=unused-import
import pdb

import matplotlib.pyplot as plt
import BayesBoom.R as R

from BayesBoom.bayesreg import (
    HierarchicalGaussianProcessRegression,
)


class TestHGP(unittest.TestCase):
    def setUp(self):
        np.random.seed(8675309)

    def tearDown(self):
        pass

    def test_nothing(self):
        pass

    def test_mcmc(self):

        group_names = ["Larry", "Curly", "Moe", "Shemp"]

        group_intercept = 4
        group_slope = 3.0
        group_data = {}
        residual_sd = 1.2

        for group in group_names:
            nobs = 20
            X = np.random.randn(nobs, 1)
            residual_sd = 7.0
            intercept = group_intercept + np.random.randn(1)
            slope = group_slope + np.random.randn(1)
            yhat = intercept + slope * X[:, 0]
            y = yhat + np.random.randn(nobs) * residual_sd
            group_data[group] = pd.DataFrame({
                "y": y,
                "x": X[:, 0],
                "group": group,
            })

        hgp = HierarchicalGaussianProcessRegression()
        full_data = pd.concat(group_data)
        hgp.add_data(predictors=full_data["x"],
                     response=full_data["y"],
                     group=full_data["group"])
        hgp.mcmc(niter=1000, ping=100)

#         pdb.set_trace()
        ### Look at the prior and a couple of the data models.  Make sure their
        ### parameter posteriors are reasonable.
        prior = hgp.prior
        print("hgp")


_debug_mode = True

if _debug_mode:
    # Turn warnings into errors.
    #    import warnings
    #    warnings.simplefilter("error")

    # Run the test you are trying to debug here.  Instantiate the test class,
    # then call the problematic test.  Call pdb.pm() in the event of an
    # exception.
    print("Hello, world!")

    rig = TestHGP()

    if hasattr(rig, "setUpClass"):
        rig.setUpClass()
    if hasattr(rig, "setUp"):
        rig.setUp()

    rig.test_mcmc()

    print("Goodbye, cruel world!")
    if False:
        pdb.set_trace()

else:
    if __name__ == "__main__":
        unittest.main(verbosity=2)
