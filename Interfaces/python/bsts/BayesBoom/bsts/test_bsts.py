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

    def tearDown(self):
        delete_if_present("bsts_llt.pkl")

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

    def test_basic_structural_model(self):
        model = Bsts()

        y = np.log(AirPassengers)
        model.add_state(LocalLinearTrendStateModel(y))
        model.add_state(SeasonalStateModel(y, nseasons=12))
        model.train(data=y, niter=1000)

        comp_fig = model.plot("comp")
        # comp_fig.show()

        predictions = model.predict(12)
        target_quantiles = [0.025] + list(np.linspace(.05, 0.95, 19)) + [0.975]
        prediction_quantiles = np.quantile(
            predictions.distribution, target_quantiles, axis=0)
        self.assertIsInstance(prediction_quantiles, np.ndarray)

        with open("bsm.pkl", "wb") as pkl:
            pickle.dump(model, pkl)

        with open("bsm.pkl", "rb") as pkl:
            m2 = pickle.load(pkl)

        pred2 = m2.predict(12)

        np.testing.assert_array_almost_equal(predictions.distribution,
                                             pred2.distribution)

        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        predictions.plot(ax=ax[0], original_series=24)
        pred_plot = pred2.plot(ax=ax[1], original_series=24)
        # fig.show()
        self.assertIsInstance(pred_plot, plt.Axes)


class TestStudentTimeSeries(unittest.TestCase):
    def setUp(self):
        n = 100
        df = 3
        random_walk = np.cumsum(np.random.randn(n) * .1)
        noise = np.random.standard_t(df=df, size=n)
        self.data = random_walk + noise

    def tearDown(self):
        delete_if_present("bsts_student_llt.pkl")

    def test_local_level(self):
        model = Bsts(family="student")
        model.add_state(LocalLevelStateModel(self.data))
        model.train(data=self.data, niter=1000)

        fname = "bsts_student_llt.pkl"
        with open(fname, "wb") as pkl:
            pickle.dump(model, pkl)

        with open(fname, "rb") as pkl:
            m2 = pickle.load(pkl)

        self.assertEqual(model.time_dimension, m2.time_dimension)
        self.assertIsInstance(m2, Bsts)

        np.testing.assert_array_almost_equal(
            model._observation_model_manager._residual_sd,
            m2._observation_model_manager._residual_sd)
        np.testing.assert_array_almost_equal(
            model._observation_model_manager._residual_df,
            m2._observation_model_manager._residual_df)

        np.testing.assert_array_almost_equal(
            model._state_models[0].state_contribution,
            m2._state_models[0].state_contribution)
        np.testing.assert_array_almost_equal(
            model._state_models[0].sigma_draws,
            m2._state_models[0].sigma_draws)

        np.testing.assert_array_almost_equal(
            model._final_state, m2._final_state)

        seed = 8675309
        pred1 = model.predict(4, seed=seed)
        pred2 = m2.predict(4, seed=seed)

        fig, ax = plt.subplots(1, 2)
        pred1.plot(ax=ax[0])
        pred2.plot(ax=ax[1])
        # fig.show()

        np.testing.assert_array_almost_equal(
            pred1.distribution,
            pred2.distribution)

    def test_basic_structural_model(self):
        model = Bsts(family="student")

        y = np.log(AirPassengers)
        model.add_state(LocalLinearTrendStateModel(y))
        model.add_state(SeasonalStateModel(y, nseasons=12))
        model.train(data=y, niter=1000)

        comp_fig = model.plot("comp")
        # comp_fig.show()

        horizon = 24
        predictions = model.predict(horizon)
        target_quantiles = [0.025] + list(np.linspace(.05, 0.95, 19)) + [0.975]
        prediction_quantiles = np.quantile(
            predictions.distribution, target_quantiles, axis=0)
        self.assertIsInstance(prediction_quantiles, np.ndarray)

        with open("bsm.pkl", "wb") as pkl:
            pickle.dump(model, pkl)

        with open("bsm.pkl", "rb") as pkl:
            m2 = pickle.load(pkl)

        pred2 = m2.predict(horizon)

        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        predictions.plot(ax=ax[0], original_series=60)
        pred_plot = pred2.plot(ax=ax[1], original_series=60)
        # fig.show()
        self.assertIsInstance(pred_plot, plt.Axes)

        np.testing.assert_array_almost_equal(predictions.distribution,
                                             pred2.distribution)


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

    rig = TestStudentTimeSeries()
    # rig = TestGaussianTimeSeries()

    if hasattr(rig, "setUpClass"):
        rig.setUpClass()
    if hasattr(rig, "setUp"):
        rig.setUp()

    # rig.test_local_level()
    rig.test_basic_structural_model()

    print("Goodbye, cruel world!")

else:
    if __name__ == "__main__":
        unittest.main(verbosity=2)
