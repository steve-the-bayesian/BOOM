import unittest
import numpy as np
import os
import pickle

from BayesBoom.bsts import (
    LocalLevelStateModel,
    LocalLinearTrendStateModel,
    SeasonalStateModel,
    SemilocalLinearTrendStateModel
)


class TestLocalLevel(unittest.TestCase):
    def setUp(self):
        np.random.seed(8675309)
        self.data = np.random.randn(100)
        self.model = LocalLevelStateModel(self.data)
        self.model.set_state_index(0)

    def tearDown(self):
        if os.path.exists("ll.pkl") and os.path.isfile("ll.pkl"):
            os.remove("ll.pkl")

    def test_state_dimension(self):
        self.assertEqual(1, self.model.state_dimension)

    def test_storage(self):
        self.model.allocate_space(10, 20)
        self.assertEqual(self.model.sigma_draws.shape, (10,))
        self.assertEqual(self.model.state_contribution.shape, (10, 20))

        state_matrix = np.zeros((4, 20))
        self.model.set_state_index(0)
        self.model._state_model.set_sigsq(9.0)

        iteration = 7
        self.model.record_state(iteration, state_matrix)
        self.assertAlmostEqual(self.model.sigma_draws[iteration], 3.0)

        self.model._state_model.set_sigsq(1.0)
        self.model.restore_state(iteration)
        self.assertAlmostEqual(self.model._state_model.sigsq, 9.0)

        with open("ll.pkl", "wb") as pkl:
            pickle.dump(self.model, pkl)

        with open("ll.pkl", "rb") as pkl:
            model = pickle.load(pkl)

        self.assertIsInstance(model, LocalLevelStateModel)

        np.testing.assert_array_equal(
            model.sigma_draws,
            self.model.sigma_draws)


class TestLocalLinearTrend(unittest.TestCase):
    def setUp(self):
        np.random.seed(8675309)
        self.data = np.random.randn(100)
        self.model = LocalLinearTrendStateModel(self.data)
        self.model.set_state_index(1)

    def tearDown(self):
        if os.path.exists("llt.pkl") and os.path.isfile("llt.pkl"):
            os.remove("llt.pkl")

    def test_state_dimension(self):
        self.assertEqual(2, self.model.state_dimension)

    def test_storage(self):
        time_dimension = 20
        niter = 10
        self.model.allocate_space(niter, time_dimension)
        self.assertEqual(self.model.sigma_level.shape, (niter,))
        self.assertEqual(self.model.sigma_slope.shape, (niter,))
        self.assertEqual(self.model.state_contribution.shape,
                         (niter, time_dimension))

        iteration = 4
        self.model.sigma_level[iteration] = 2.6
        self.model.sigma_slope[iteration] = 1.9
        self.model.restore_state(iteration)
        self.assertAlmostEqual(self.model._state_model.sigma_level, 2.6)
        self.assertAlmostEqual(self.model._state_model.sigma_slope, 1.9)

        state_matrix = np.random.randn(3, time_dimension)
        iteration = 7
        self.model.record_state(iteration, state_matrix)
        self.assertAlmostEqual(self.model.sigma_level[iteration], 2.6)
        self.assertAlmostEqual(self.model.sigma_slope[iteration], 1.9)
        np.testing.assert_array_equal(
            self.model._state_contribution[iteration, :],
            state_matrix[1, :])

        with open("llt.pkl", "wb") as pkl:
            pickle.dump(self.model, pkl)

        with open("llt.pkl", "rb") as pkl:
            m2 = pickle.load(pkl)

        np.testing.assert_array_equal(self.model.sigma_level, m2.sigma_level)
        np.testing.assert_array_equal(self.model.sigma_slope, m2.sigma_slope)


class TestSeasonalStateModel(unittest.TestCase):
    def setUp(self):
        np.random.seed(8675309)
        self.data = np.random.randn(100)
        self.model = SeasonalStateModel(self.data, nseasons=4)
        self.model.set_state_index(1)

    def tearDown(self):
        if os.path.exists("seas.pkl") and os.path.isfile("seas.pkl"):
            os.remove("seas.pkl")

    def test_state_dimension(self):
        self.assertEqual(3, self.model.state_dimension)

    def test_storage(self):
        time_dimension = 20
        niter = 10
        self.model.allocate_space(niter, time_dimension)
        self.assertEqual(self.model.sigma_draws.shape, (niter, ))
        self.assertEqual(self.model._state_contribution.shape,
                         (niter, time_dimension))

        self.model.set_state_index(2)
        state_matrix = np.random.randn(3, time_dimension)
        iteration = 4
        self.model.sigma_draws[4] = 1.9
        self.model.restore_state(iteration)
        self.assertAlmostEqual(1.9, self.model._state_model.sigma)

        iteration = 8
        self.model.record_state(iteration, state_matrix)
        self.assertAlmostEqual(self.model.sigma_draws[iteration], 1.9)
        np.testing.assert_array_equal(
            self.model._state_contribution[iteration, :],
            state_matrix[self.model._state_index, :])

        with open("seas.pkl", "wb") as pkl:
            pickle.dump(self.model, pkl)

        with open("seas.pkl", "rb") as pkl:
            _ = pickle.load(pkl)


class TestSemilocalLinearTrendStateModel(unittest.TestCase):
    def setUp(self):
        np.random.seed(8675309)
        self.data = np.random.randn(100)
        self.model = SemilocalLinearTrendStateModel(self.data)
        self.model.set_state_index(0)

    def tearDown(self):
        if os.path.exists("sllt.pkl") and os.path.isfile("sllt.pkl"):
            os.remove("sllt.pkl")

    def test_state_dimension(self):
        self.assertEqual(self.model.state_dimension, 3)

    def test_storage(self):
        time_dimension = 20
        niter = 10
        self.model.allocate_space(niter, time_dimension)
        self.assertEqual(self.model.level_sigma.shape, (niter,))
        self.assertEqual(self.model.slope_sigma.shape, (niter,))
        self.assertEqual(self.model.slope_ar1.shape, (niter,))
        self.assertEqual(self.model.slope_mean.shape, (niter,))
        self.assertEqual(self.model.state_contribution.shape,
                         (niter, time_dimension))

        iteration = 4
        self.model.level_sigma[iteration] = 2.6
        self.model.slope_sigma[iteration] = 1.9
        self.model.slope_mean[iteration] = -7.2
        self.model.slope_ar1[iteration] = 0.8
        self.model.restore_state(iteration)
        self.assertAlmostEqual(self.model._state_model.level_sd, 2.6)
        self.assertAlmostEqual(self.model._state_model.slope_sd, 1.9)
        self.assertAlmostEqual(self.model._state_model.slope_mean, -7.2)
        self.assertAlmostEqual(self.model._state_model.slope_ar_coefficient,
                               0.8)

        state_matrix = np.random.randn(3, time_dimension)
        iteration = 7
        self.model.record_state(iteration, state_matrix)
        self.assertAlmostEqual(self.model.level_sigma[iteration], 2.6)
        self.assertAlmostEqual(self.model.slope_sigma[iteration], 1.9)
        self.assertAlmostEqual(self.model.slope_mean[iteration], -7.2)
        self.assertAlmostEqual(self.model.slope_ar1[iteration], 0.8)
        np.testing.assert_array_equal(
            self.model._state_contribution[iteration, :],
            state_matrix[self.model._state_index, :])

        with open("sllt.pkl", "wb") as pkl:
            pickle.dump(self.model, pkl)

        with open("sllt.pkl", "rb") as pkl:
            m2 = pickle.load(pkl)

        np.testing.assert_array_equal(self.model.level_sigma, m2.level_sigma)
        np.testing.assert_array_equal(self.model.slope_sigma, m2.slope_sigma)
        np.testing.assert_array_equal(self.model.slope_ar1, m2.slope_ar1)
        np.testing.assert_array_equal(self.model.slope_mean, m2.slope_mean)


_debug_mode = False

if _debug_mode:
    import pdb  # noqa

    # Turn warnings into errors.
    # warnings.simplefilter("error")

    # Run the test you are trying to debug here.  Instantiate the test class,
    # then call the problematic test.  Call pdb.pm() in the event of an
    # exception.
    print("Hello, world!")
    rig = TestSemilocalLinearTrendStateModel()
    if hasattr(rig, "setUpClass"):
        rig.setUpClass()
    if hasattr(rig, "setUp"):
        rig.setUp()

    rig.test_storage()

    print("Goodbye, cruel world!")

else:
    if __name__ == "__main__":
        unittest.main(verbosity=2)
