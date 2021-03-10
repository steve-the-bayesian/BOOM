import unittest
import numpy as np
import pandas as pd
import pickle
import json

from BayesBoom.bsts import (
    Bsts,
    AirPassengers,
    LocalLevelStateModel,
    LocalLinearTrendStateModel,
    SeasonalStateModel,
)


class TestGaussianTimeSeries(unittest.TestCase):
    def setUp(self):
        n = 100
        random_walk = np.cumsum(np.random.randn(n) * .1)
        noise = np.random.randn(n)
        self.data = random_walk + noise

    def test_local_level(self):
        model = Bsts()
        model.add_state(LocalLevelStateModel(self.data))
        model.train(data=self.data, niter=10)

        with open("bsts_llt.pkl", "wb") as pkl:
            pickle.dump(model, pkl)

        with open("bsts_llt.pkl", "rb") as pkl:
            m2 = pickle.load(pkl)

        self.assertIsInstance(m2, Bsts)
        np.testing.assert_array_equal(m2._final_state, model._final_state)

    def test_seasonal(self):
        model = Bsts()

        y = np.log(AirPassengers)
        model.add_state(LocalLinearTrendStateModel(y))
        model.add_state(SeasonalStateModel(y, nseasons=12))
        model.train(data=y, niter=1000)

        import matplotlib.pyplot as plt
        model.plot("comp")
        plt.show()

        predictions = model.predict(12)
        target_quantiles = [0.025] + list(np.linspace(.05, 0.95, 19)) + [0.975]
        prediction_quantiles = np.quantile(
            predictions.distribution, target_quantiles, axis=0)


# class TestStateSpaceRegression(unittest.TestCase):

#     def test_bsts_regression(self):
#         sample_size = 100
#         random_walk = np.cumsum(np.random.randn(sample_size) * .05)
#         noise = np.random.randn(sample_size) * .2
#         predictors = pd.DataFrame(np.random.randn(sample_size, 2),
#                                   columns=["x1", "x2"])
#         y = random_walk + 2.3 * predictors["x1"] -1.8 * predictors["x2"] + noise
#         predictors["y"] = y

#         model = Bsts()
#         model.add_state(LocalLevelStateModel(y))
#         model.train(data=predictors, formula="y ~ x1 + x2", niter=100)

#         with open("bsts_reg.pkl", "wb") as pkl:
#             pickle.dump(model, pkl)

#         with open("bsts_reg.pkl", "rb") as pkl:
#             m2 = pickle.load(pkl)



_debug_mode = False

if _debug_mode:
    import pdb  # noqa

    # Turn warnings into errors.
    # warnings.simplefilter("error")

    # Run the test you are trying to debug here.  Instantiate the test class,
    # then call the problematic test.  Call pdb.pm() in the event of an
    # exception.
    print("Hello, world!")
    rig = TestGaussianTimeSeries()
    if hasattr(rig, "setUpClass"):
        rig.setUpClass()
    if hasattr(rig, "setUp"):
        rig.setUp()

    rig.test_seasonal()

    print("Goodbye, cruel world!")

else:
    if __name__ == "__main__":
        unittest.main(verbosity=2)
