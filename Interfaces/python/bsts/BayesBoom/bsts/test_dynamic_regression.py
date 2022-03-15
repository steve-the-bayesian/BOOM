import unittest
import numpy as np
import pandas as pd
import pickle
import json

import pdb

import matplotlib.pyplot as plt

from BayesBoom.R import delete_if_present

from BayesBoom.bsts import (
    Bsts,
    AirPassengers,
    DynamicRegressionStateModel,
    LocalLevelStateModel,
    LocalLinearTrendStateModel,
    SeasonalStateModel,
)


def default_colnames(n):
    return ["X" + str(i + 1) for i in range(n)]


class TestDynamicRegression(unittest.TestCase):
    def setUp(self):
        n = 100
        xdim = 3
        coefficients = np.cumsum(np.random.randn(n, xdim) * .1, axis=0)
        self._predictors = np.random.randn(n, xdim)
        self._random_walk = np.cumsum(np.random.randn(n) * .1)
        self._reg = np.sum(coefficients * self._predictors, axis=1)
        self._noise = np.random.randn(n)
        self._y = self._random_walk + self._reg + self._noise

    def tearDown(self):
        pass
        # delete_if_present("bsts_llt.pkl")

    def test_local_level(self):
        model = Bsts()
        model.add_state(LocalLevelStateModel(self._y))
        data = pd.DataFrame(
            self._predictors,
            columns=default_colnames(self._predictors.shape[1]))
        data["y"] = self._y

        model.add_state(DynamicRegressionStateModel(
            "y ~ X1 + X2 + X3", data=data))

        model.train(data=data["y"], niter=1000)

    def test_plots(self):
        pass

    def test_predictions(self):
        pass

    def test_serialization(self):
        pass


_debug_mode = False


if _debug_mode:
    import pdb  # noqa

    # Turn warnings into errors.
    # warnings.simplefilter("error")

    # Run the test you are trying to debug here.  Instantiate the test class,
    # then call the problematic test.  Call pdb.pm() in the event of an
    # exception.
    print("Hello, world!")

    rig = TestDynamicRegression()
    # rig = TestGaussianTimeSeries()

    if hasattr(rig, "setUpClass"):
        rig.setUpClass()
    if hasattr(rig, "setUp"):
        rig.setUp()

    rig.test_local_level()

    print("Goodbye, cruel world!")

else:
    if __name__ == "__main__":
        unittest.main(verbosity=2)
