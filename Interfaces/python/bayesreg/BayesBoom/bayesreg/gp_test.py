import unittest
import numpy as np
import pandas as pd
import pickle

# pylint: disable=unused-import
import pdb

import matplotlib.pyplot as plt

import BayesBoom.R as R

from BayesBoom.bayesreg import (
    MahalanobisKernel,
    ZeroFunction,
    GaussianProcessRegression,
)


class TestGaussianProcessRegression(unittest.TestCase):
    def setUp(self):
        np.random.seed(8675309)

    def tearDown(self):
        pass

    def test_nothing(self):
        pass

    def test_mcmc(self):
        nobs = 20
        X = np.random.randn(nobs, 1)
        residual_sd = 7.0
        intercept = 4
        y = intercept + 3 * X[:, 0] + np.random.randn(nobs) * residual_sd

        mean_function = ZeroFunction()
        kernel = MahalanobisKernel(X, scale_prior = R.SdPrior(1, 1))
        model = GaussianProcessRegression(
            mean_function,
            kernel,
            100.0)

        model.set_prior(R.SdPrior(7.0, 1.0))
        model.add_data(X, y)
        model.mcmc(20, ping=5)


_debug_mode = False

if _debug_mode:
    import pdb  # noqa

    # Turn warnings into errors.
#    import warnings
#    warnings.simplefilter("error")

    # Run the test you are trying to debug here.  Instantiate the test class,
    # then call the problematic test.  Call pdb.pm() in the event of an
    # exception.
    print("Hello, world!")

    rig = TestGaussianProcessRegression()

    if hasattr(rig, "setUpClass"):
        rig.setUpClass()
    if hasattr(rig, "setUp"):
        rig.setUp()

    rig.test_mcmc()

    print("Goodbye, cruel world!")

else:
    if __name__ == "__main__":
        unittest.main(verbosity=2)
