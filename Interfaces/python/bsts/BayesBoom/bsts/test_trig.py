import unittest
import numpy as np
import pickle
import pandas as pd   # noqa
import json           # noqa

# pylint: disable=unused-import
import pdb

import matplotlib.pyplot as plt    # noqa

from BayesBoom.R import delete_if_present

from BayesBoom.bsts import (
    Bsts,
    AirPassengers,
    LocalLinearTrendStateModel,
    TrigStateModel,
)


class TestTrig(unittest.TestCase):
    def setUp(self):
        self._y = np.log(AirPassengers)

    def tearDown(self):
        delete_if_present("trig.pkl")

    def test_local_level(self):
        model = Bsts()
        model.add_state(LocalLinearTrendStateModel(self._y))
        model.add_state(TrigStateModel(self._y, period=12, frequencies=[1, 2]))
        model.train(data=self._y, niter=1000)

        fname = "trig.pkl"
        with open(fname, "wb") as pkl:
            pickle.dump(model, pkl)

        with open(fname, "rb") as pkl:
            m2 = pickle.load(pkl)

        self.assertEqual(model.time_dimension, m2.time_dimension)
        self.assertIsInstance(m2, Bsts)

    def test_predictions(self):
        pass


_debug_mode = False


if _debug_mode:
    # Turn warnings into errors.
    # warnings.simplefilter("error")

    # Run the test you are trying to debug here.  Instantiate the test class,
    # then call the problematic test.  Call pdb.pm() in the event of an
    # exception.
    print("Hello, world!")

    rig = TestTrig()

    if hasattr(rig, "setUpClass"):
        rig.setUpClass()
    if hasattr(rig, "setUp"):
        rig.setUp()

    rig.test_local_level()

    print("Goodbye, cruel world!")

else:
    if __name__ == "__main__":
        unittest.main(verbosity=2)
