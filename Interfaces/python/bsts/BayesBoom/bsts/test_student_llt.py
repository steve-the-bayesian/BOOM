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
    StudentLocalLinearTrendStateModel,
    simulate_student_local_linear_trend
)


def default_colnames(n):
    return ["X" + str(i + 1) for i in range(n)]


class TestStudentLocalLinearTrend(unittest.TestCase):
    def setUp(self):
        sample_size = 100
        self._trend = simulate_student_local_linear_trend(nsteps=sample_size)
        self._y = self._trend + np.random.randn(sample_size)

    def tearDown(self):
        delete_if_present("student_llt.pkl")

    def test_local_level(self):
        model = Bsts()
        model.add_state(StudentLocalLinearTrendStateModel(self._y))
        model.train(data=self._y, niter=1000)

        fname = "student_llt.pkl"
        with open(fname, "wb") as pkl:
            pickle.dump(model, pkl)

        with open(fname, "rb") as pkl:
            m2 = pickle.load(pkl)

        self.assertEqual(model.time_dimension, m2.time_dimension)
        self.assertIsInstance(m2, Bsts)



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
