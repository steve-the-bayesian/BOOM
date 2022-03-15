import unittest
import numpy as np
import pandas as pd
import pickle

import pdb

import matplotlib.pyplot as plt

import BayesBoom.R as R

from BayesBoom.spikeslab import dot

from BayesBoom.bsts import (
    Bsts,
    AirPassengers,
    compare_bsts_models,
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
        R.delete_if_present("bsts_llt.pkl")

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

        errors = model.one_step_prediction_errors()
        self.assertIsInstance(errors, np.ndarray)
        self.assertEqual(errors.shape, (10, 100))

        errors = model.one_step_prediction_errors(burn=7)
        self.assertEqual(errors.shape, (3, 100))

        foo = model._model.simulate_holdout_prediction_errors(
            50, 90, False).to_numpy()
        self.assertEqual(foo.shape, (50, 100))
        self.assertEqual(100, model.time_dimension)

        cutpoints = [60, 80, 100]
        errors = model.one_step_prediction_errors(cutpoints=cutpoints)
        self.assertIsInstance(errors, dict)
        self.assertEqual(len(errors), len(cutpoints))
        self.assertEqual(errors[60].shape,  (10, 100))
        self.assertEqual(errors[80].shape,  (10, 100))
        self.assertEqual(errors[100].shape,  (10, 100))
        self.assertEqual(model.time_dimension, 100)


    def test_basic_structural_model(self):
        model = Bsts()

        y = np.log(AirPassengers)
        model.add_state(LocalLinearTrendStateModel(y))
        model.add_state(SeasonalStateModel(y, nseasons=12))
        model.train(data=y, niter=1000)

        fig = model.plot("comp")
        if _show_figs:
            fig.show()

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
        pred_plot_fig, pred_plot_ax = pred2.plot(ax=ax[1], original_series=24)
        if _show_figs:
            fig.show()
        self.assertIsInstance(pred_plot_ax, plt.Axes)


class TestGaussianRegression(unittest.TestCase):
    def setUp(self):
        np.random.seed(8675309)
        n = 100
        p = 10
        pgood = 3
        X = np.random.randn(n, p)
        beta = np.zeros(p)
        beta[:pgood] = np.random.randn(pgood)
        yhat = X @ beta

        random_walk = np.cumsum(np.random.randn(n) * .1)
        noise = np.random.randn(n) * .05
        y = yhat + random_walk + noise
        self.data = pd.DataFrame(X, columns=R.paste0("X", np.arange(1, p + 1)))
        self.data["y"] = y

    def test_state_space_regression(self):
        pass


class TestStudentTimeSeries(unittest.TestCase):
    def setUp(self):
        np.random.seed(8675309)
        n = 100
        df = 3
        random_walk = np.cumsum(np.random.randn(n) * .1)
        noise = np.random.standard_t(df=df, size=n)
        self.data = random_walk + noise

    def tearDown(self):
        R.delete_if_present("bsts_student_llt.pkl")
        R.delete_if_present("bsm.pkl")

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
        if _show_figs:
            fig.show()

        np.testing.assert_array_almost_equal(
            pred1.distribution,
            pred2.distribution)

    def test_basic_structural_model(self):
        model = Bsts(family="student")

        y = np.log(AirPassengers)
        model.add_state(LocalLinearTrendStateModel(y))
        model.add_state(SeasonalStateModel(y, nseasons=12))
        model.train(data=y, niter=1000)

        fig = model.plot("comp")
        if _show_figs:
            fig.show()

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
        pred_plot_fig, pred_plot_ax = pred2.plot(ax=ax[1], original_series=60)
        if _show_figs:
            fig.show()
        self.assertIsInstance(pred_plot_ax, plt.Axes)

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


class TestPlots(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        np.random.seed(8675309)
        np.random.seed(0xDEADBEEF)
        fake_data = np.random.randn(100, 10)
        fake_coefficients = np.array([5, -4, 3] + [0] * 7)
        random_walk = np.cumsum(np.random.randn(100))
        noise = np.random.randn(100) * 1.5
        y = random_walk + fake_data @ fake_coefficients + noise
        data = pd.DataFrame(
            fake_data,
            columns=["V" + str(x+1) for x in range(10)],
        )
        data["y"] = y
        model = Bsts()
        model.add_state(LocalLevelStateModel(y))
        model.train(formula=f"y ~ {dot(data, omit='y')}", niter=100, data=data)
        self._regression_model = model

        data = np.log(AirPassengers)
        model = Bsts()
        model.add_state(LocalLinearTrendStateModel(data))
        model.add_state(SeasonalStateModel(data, nseasons=12))
        model.train(data, niter=100)
        self._model = model

        model = Bsts()
        model.add_state(LocalLinearTrendStateModel(data))
        model.train(data, niter=100)
        self._model2 = model

    def test_plot_state(self):
        _, ax = plt.subplots()
        foo, bar = self._model.plot(ax=ax)
        self.assertIsInstance(bar, plt.Axes)

        _, ax = plt.subplots()
        foo, bar = self._regression_model.plot(ax=ax)
        self.assertIsInstance(bar, plt.Axes)

    def test_plot_components(self):
        fig = plt.figure()
        foo = self._model.plot("comp", fig=fig)
        self.assertIsInstance(foo, plt.Figure)

    def test_plot_seasonal(self):
        fig = plt.figure()
        foo = self._model.plot("seas", fig=fig)
        self.assertIsInstance(foo, plt.Figure)

    def test_plot_residuals(self):
        fig, ax = plt.subplots()
        foo = self._model.plot_residuals(ax=ax)
        self.assertIsInstance(foo, plt.Axes)

        fig, ax = plt.subplots()
        foo = self._regression_model.plot_residuals(ax=ax)
        self.assertIsInstance(foo, plt.Axes)

        if _show_figs:
            fig.show()

    def test_plot_forecast_distribution(self):
        fig = self._model.plot("forecast_distribution", cex_actuals=1)
        self.assertIsInstance(fig, plt.Figure)
        if _show_figs:
            fig.show()

        fig2 = self._model.plot("forecast_distribution",
                                cutpoints=(50, 80, 100))
        if _show_figs:
            fig2.show()

    def test_plot_size(self):
        fig, ax = plt.subplots()
        foo = self._regression_model.plot_size(ax=ax)
        if _show_figs:
            fig.show()
        self.assertIsInstance(foo, plt.Axes)

    def test_plot_predictors(self):
        fig, ax = plt.subplots()
        foo = self._regression_model.plot(
            "predictors", ax=ax, short_names=False)
        if _show_figs:
            fig.show()
        self.assertIsInstance(foo, plt.Axes)

    def test_compare_bsts_models(self):
        fig = plt.figure()
        models = {
            "bsm": self._model,
            "trend only": self._model2
        }
        ans = compare_bsts_models(models, fig=fig)
        self.assertIsInstance(ans, plt.Figure)


_debug_mode = True
_show_figs = _debug_mode

if _debug_mode:
    import pdb  # noqa

    # Turn warnings into errors.
#    import warnings
#    warnings.simplefilter("error")

    # Run the test you are trying to debug here.  Instantiate the test class,
    # then call the problematic test.  Call pdb.pm() in the event of an
    # exception.
    print("Hello, world!")

    rig = TestPlots()

    if hasattr(rig, "setUpClass"):
        rig.setUpClass()
    if hasattr(rig, "setUp"):
        rig.setUp()

    rig.test_plot_forecast_distribution()

    print("Goodbye, cruel world!")

else:
    if __name__ == "__main__":
        unittest.main(verbosity=2)
