import unittest
import numpy as np
import pandas as pd
import pickle
import json

import pdb

import matplotlib.pyplot as plt

from BayesBoom.bsts import (
    Bsts,
    AirPassengers,
    ArStateModel,
    AutoArStateModel,
    LocalLinearTrendStateModel,
    SeasonalStateModel,
)


class TestAr(unittest.TestCase):
    def setUp(self):
        self._y = np.log(AirPassengers)

    def tearDown(self):
        pass
        # delete_if_present("bsts_llt.pkl")

    def test_basic_ar(self):
        model = Bsts()
        model.add_state(LocalLinearTrendStateModel(self._y))
        model.add_state(SeasonalStateModel(self._y, nseasons=12))
        model.add_state(ArStateModel(self._y, lags=1))
        model.train(data=self._y, niter=1000)

    def test_auto_ar(self):
        model = Bsts()
        model.add_state(LocalLinearTrendStateModel(self._y))
        model.add_state(SeasonalStateModel(self._y, nseasons=12))
        model.add_state(AutoArStateModel(self._y, lags=1))
        model.train(data=self._y, niter=1000)

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

    rig = TestAr()

    if hasattr(rig, "setUpClass"):
        rig.setUpClass()
    if hasattr(rig, "setUp"):
        rig.setUp()

    rig.test_basic_ar()

    print("Goodbye, cruel world!")

else:
    if __name__ == "__main__":
        unittest.main(verbosity=2)
