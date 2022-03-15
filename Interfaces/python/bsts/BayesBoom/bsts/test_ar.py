import unittest
import numpy as np
import pickle


from BayesBoom.bsts import (
    Bsts,
    AirPassengers,
    ArStateModel,
    AutoArStateModel,
    LocalLinearTrendStateModel,
    SeasonalStateModel,
)

from BayesBoom.R import delete_if_present


class TestAr(unittest.TestCase):
    def setUp(self):
        self._y = np.log(AirPassengers)

    def tearDown(self):
        delete_if_present("ar.pkl")

    def test_basic_ar(self):
        model = Bsts()
        model.add_state(LocalLinearTrendStateModel(self._y))
        model.add_state(SeasonalStateModel(self._y, nseasons=12))
        model.add_state(ArStateModel(self._y, lags=1))
        model.train(data=self._y, niter=1000)

        fname = "ar.pkl"
        with open(fname, "wb") as pkl:
            pickle.dump(model, pkl)

        with open(fname, "rb") as pkl:
            m2 = pickle.load(pkl)

        self.assertEqual(model.time_dimension, m2.time_dimension)
        self.assertIsInstance(m2, Bsts)

        pred = model.predict(12)

    def test_auto_ar(self):
        model = Bsts()
        model.add_state(LocalLinearTrendStateModel(self._y))
        model.add_state(SeasonalStateModel(self._y, nseasons=12))
        model.add_state(AutoArStateModel(self._y, lags=1))
        model.train(data=self._y, niter=1000)

        fname = "ar.pkl"
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
