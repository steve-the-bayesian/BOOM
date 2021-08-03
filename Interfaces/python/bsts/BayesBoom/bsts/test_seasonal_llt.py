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
    LocalLinearTrendStateModel,
    GeneralSeasonalLLT,
)


def default_colnames(n):
    return ["X" + str(i + 1) for i in range(n)]


class TestSeasonalLLT(unittest.TestCase):
    def setUp(self):
        np.random.seed(8675309)

    def tearDown(self):
        delete_if_present("seasonal_llt.pkl")

    def test_mcmc(self):
        model = Bsts()
        model.add_state(LocalLinearTrendStateModel(AirPassengers))
        model.add_state(GeneralSeasonalLLT(AirPassengers, nseasons=12))

        model.train(data=AirPassengers, niter=1000)

        fname = "seasonal_llt.pkl"
        with open(fname, "wb") as pkl:
            pickle.dump(model, pkl)

        with open(fname, "rb") as pkl:
            m2 = pickle.load(pkl)

        self.assertEqual(model.time_dimension, m2.time_dimension)
        self.assertIsInstance(m2, Bsts)

        fig = model.plot("comp")
        fig.show()

        pred = model.predict(12)
        fig2, ax = pred.plot()
        fig2.show()

    def test_plots(self):
        pass

    def test_predictions(self):
        pass

    def test_serialization(self):
        pass


_debug_mode = True


if _debug_mode:
    import pdb  # noqa

    # Turn warnings into errors.
    # warnings.simplefilter("error")

    # Run the test you are trying to debug here.  Instantiate the test class,
    # then call the problematic test.  Call pdb.pm() in the event of an
    # exception.
    print("Hello, world!")

    rig = TestSeasonalLLT()

    if hasattr(rig, "setUpClass"):
        rig.setUpClass()
    if hasattr(rig, "setUp"):
        rig.setUp()

    rig.test_mcmc()

    print("Goodbye, cruel world!")

else:
    if __name__ == "__main__":
        unittest.main(verbosity=2)
