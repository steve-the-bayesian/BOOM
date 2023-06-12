import unittest

from BayesBoom.spikeslab import (
    dot,
    lm_spike,
    RegressionSpikeSlabPrior,
    StudentSpikeSlabPrior,
    BigAssSpikeSlab
)
from BayesBoom.R import delete_if_present

import numpy as np
import pandas as pd
import scipy.sparse
import pickle


class SpikeSlabTest(unittest.TestCase):
    def setUp(self):
        np.random.seed(8675309)

    def tearDown(self):
        delete_if_present("prior.pkl")

    def test_mcmc(self):
        # Run the test with a big sample size and a small residual SD, so
        # you'll be sure to find the right answer.
        sample_size = 10000
        ngood = 5
        nbad = 30
        niter = 250
        x = np.random.randn(sample_size, ngood + nbad)

        beta = np.random.randn(ngood) * 4

        b0 = 7.2
        residual_sd = .3
        yhat = b0 + x[:, :ngood] @ beta
        errors = np.random.randn(sample_size) * residual_sd
        y = yhat + errors

        data = pd.DataFrame(
            x, columns=["X" + str(i) for i in range(x.shape[1])])
        data["y"] = y
        formula = "y ~ " + dot(data, ["y"])

        model = lm_spike(formula, niter=niter, data=data)
        self.assertTrue(isinstance(
            model._coefficient_draws,
            scipy.sparse.csc_matrix)
        )
        self.assertEqual(model._coefficient_draws.shape[0], niter)
        self.assertEqual(model._coefficient_draws.shape[1], ngood + nbad + 1)

        self.assertEqual(len(model._residual_sd), niter)
        self.assertTrue(np.all(model._residual_sd > 0))

        # Test the read-only properties
        self.assertEqual(len(model._log_likelihood), niter)
        self.assertEqual(model.xdim, 1 + ngood + nbad)

        # Test the predict method.
        pred = model.predict(data, burn=3)
        self.assertTrue(isinstance(pred, np.ndarray))
        self.assertEqual(len(pred.shape), 2)
        self.assertEqual(pred.shape[0], niter-3)
        self.assertEqual(pred.shape[1], sample_size)

    def test_dot(self):
        X = pd.DataFrame(np.random.randn(10, 3), columns=["X1", "X2", "X3"])

        # Test with an empty omit.
        formula = dot(X)
        self.assertEqual(formula, "(X1+X2+X3)")

        X["y"] = 7.0
        formula = "y ~ " + dot(X, omit=["y"])
        self.assertEqual(formula, "y ~ (X1+X2+X3)")

        # Test that omit still works as a string.
        formula = "y ~ " + dot(X, omit="y")
        self.assertEqual(formula, "y ~ (X1+X2+X3)")

        # Test with multiple omitted columns.
        formula = "y ~ " + dot(X, omit=["y", "X2"])
        self.assertEqual(formula, "y ~ (X1+X3)")

    def test_regression_spike_slab_prior(self):
        n = 10
        X = pd.DataFrame(np.random.randn(n, 3), columns=["X1", "X2", "X3"])
        y = np.random.randn(n)
        prior = RegressionSpikeSlabPrior(X.values, y)
        self.assertIsInstance(prior, RegressionSpikeSlabPrior)

        fname = "prior.pkl"
        with open(fname, "wb") as pkl:
            pickle.dump(prior, pkl)

        with open(fname, "rb") as pkl:
            p2 = pickle.load(pkl)

        self.assertEqual(set(prior.__dict__.keys()),
                         set(p2.__dict__.keys()))

    def test_student_spike_slab_prior(self):
        n = 10
        X = pd.DataFrame(np.random.randn(n, 3), columns=["X1", "X2", "X3"])
        y = np.random.randn(n)
        prior = StudentSpikeSlabPrior(X.values, y)

        fname = "prior.pkl"
        with open(fname, "wb") as pkl:
            pickle.dump(prior, pkl)

        with open(fname, "rb") as pkl:
            p2 = pickle.load(pkl)

        self.assertEqual(set(prior.__dict__.keys()),
                         set(p2.__dict__.keys()))


class BigAssSpikeSlabTest(unittest.TestCase):
    def setUp(self):
        np.random.seed(8675309)

    def test_mcmc(self):
        nobs = 100000
        dim = 1000
        X = np.random.randn(nobs, dim)
        X[:, 0] = 1.0
        coefficients = np.zeros(dim)
        coefficients[0] = 28
        coefficients[3] = -72
        coefficients[84] = 54
        coefficients[93] = 180

        residual_sd = .3
        yhat = X @ coefficients
        y = yhat + np.random.randn(nobs) * residual_sd

        model = BigAssSpikeSlab(dim, subordinate_model_max_dim=50)
        i = 0
        chunk_size = 1000
        while (i < nobs):
            chunk = range(i, min(i + chunk_size, nobs))
            model.stream_data_for_initial_screen(X[chunk, :], y[chunk])
            i += chunk_size
        model.initial_screen()


_debug_mode = True

if _debug_mode:
    import pdb  # noqa

    # Turn warnings into errors.
    # warnings.simplefilter("error")

    # Run the test you are trying to debug here.  Instantiate the test class,
    # then call the problematic test.  Call pdb.pm() in the event of an
    # exception.
    print("Hello, world!")

    rig = BigAssSpikeSlabTest()

    if hasattr(rig, "setUpClass"):
        rig.setUpClass()
    if hasattr(rig, "setUp"):
        rig.setUp()

    rig.test_mcmc()

    print("Goodbye, cruel world!")

if __name__ == "__main__":
    unittest.main()
