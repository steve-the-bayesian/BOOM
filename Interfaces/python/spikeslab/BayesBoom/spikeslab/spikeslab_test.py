import unittest

from BayesBoom.spikeslab import (
    dot,
    lm_spike,
    lm_spike_summary,
    RegressionSpikeSlabPrior,
    StudentSpikeSlabPrior,
    BigAssSpikeSlab
)

from BayesBoom.R import delete_if_present
import BayesBoom.R as R

import numpy as np
import pandas as pd
import scipy.sparse
import pickle

import matplotlib.pyplot as plt


def write_R_vector(v):
    if v.size == 0:
        return "numeric(0)"
    else:
        ans = "c(" + str(v[0])
        for el in v[1:]:
            ans += ", " + str(el)
        return ans + ")"


def write_R_matrix(m):
    if m.size == 0:
        return "numeric(0)"
    else:
        ans = "matrix(c("
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                if i > 0 or j > 0:
                    ans += ", "
                ans += str(m[i, j])
        ans += "), nrow = " + str(m.shape[0])
        ans += ", byrow = TRUE)"
        return ans


class SpikeSlabTest(unittest.TestCase):
    def setUp(self):
        np.random.seed(8675309)

    def tearDown(self):
        delete_if_present("prior.pkl")

    def simulate_data(self, sample_size=10000, ngood=4, nbad=30,
                      residual_sd=.3):
        x = np.random.randn(sample_size, ngood + nbad)
        self._beta = np.random.randn(ngood) * 4
        self._ngood = ngood
        self._nbad = nbad
        self._sample_size = sample_size

        self._b0 = 7.2
        residual_sd = .3
        yhat = self._b0 + x[:, :ngood] @ self._beta
        errors = np.random.randn(sample_size) * residual_sd
        y = yhat + errors
        self._data = pd.DataFrame(
            x, columns=["X" + str(i) for i in range(x.shape[1])])
        self._data["y"] = y

    def test_mcmc(self):
        # Run the test with a big sample size and a small residual SD, so
        # you'll be sure to find the right answer.
        self.simulate_data()
        formula = "y ~ " + dot(self._data, ["y"])
        niter = 1000

        model = lm_spike(formula, niter=niter, data=self._data)
        self.assertTrue(isinstance(
            model._coefficient_draws,
            scipy.sparse.csc_matrix)
        )
        self.assertEqual(model._coefficient_draws.shape[0], niter)
        self.assertEqual(model._coefficient_draws.shape[1],
                         self._ngood + self._nbad + 1)

        self.assertEqual(len(model._residual_sd), niter)
        self.assertTrue(np.all(model._residual_sd > 0))

        # Test the read-only properties
        self.assertEqual(len(model._log_likelihood), niter)
        self.assertEqual(model.xdim, 1 + self._ngood + self._nbad)

        # Test the predict method.
        pred = model.predict(self._data, burn=3)
        self.assertTrue(isinstance(pred, np.ndarray))
        self.assertEqual(len(pred.shape), 2)
        self.assertEqual(pred.shape[0], niter-3)
        self.assertEqual(pred.shape[1], self._data.shape[0])

    def test_mcmc_with_zero_variance_predictor(self):
        self.simulate_data()
        self._data["zero_variance"] = 3.14
        formula = "y ~ " + dot(self._data, ["y"])
        niter = 1000
        model = lm_spike(formula, niter=niter, data=self._data)
        self.assertIsInstance(model, lm_spike)

    def test_mcmc_from_suf(self):
        niter = 1000
        self.simulate_data()
        y = self._data["y"].values
        x = self._data.drop("y", axis=1).values
        intercept = np.ones((self._sample_size, 1))
        x = np.concatenate((intercept, x), axis=1)

        xtx = x.T @ x
        xty = x.T @ y
        x1 = x[:, 1]
        self.assertAlmostEqual(np.sum(x1 * y),  xty[1])
        x2 = x[:, 2]
        self.assertAlmostEqual(np.sum(x1 * x2), xtx[1, 2])

        X = pd.DataFrame(x[:, 1:],
                         columns=["X" + str(i) for i in range(1, x.shape[1])])
        xnames = ["Intercept"] + [str(name) for name in X.columns]
        X["y"] = y

        sample_sd = np.std(y, ddof=1)
        xbar = np.mean(x, axis=0)
        suf = R.RegSuf(xtx,
                       xty,
                       sample_sd=sample_sd,
                       sample_size=self._sample_size,
                       xbar=xbar)
        model = lm_spike(formula=None, data=suf, niter=niter, xnames=xnames)

        raw_model = lm_spike("y ~ " + dot(X, "y"), niter=niter, data=X)
        self.assertEqual(len(model._log_likelihood), niter)

        suf_summary = model.summary(burn=10)
        self.assertTrue(isinstance(suf_summary, lm_spike_summary))

        raw_summary = raw_model.summary(burn=10)

        # Check that the r-squares from the raw and suf models agree closely.

        pct_change_r2 = (raw_summary.r2 - suf_summary.r2) / suf_summary.r2
        self.assertLess(np.abs(pct_change_r2), .01)

        # Check that the residual standard deviations from the raw and suf
        # models agree closely.
        pct_change_sigma = (
            (raw_summary.residual_sd - suf_summary.residual_sd) /
            suf_summary.residual_sd
        )
        self.assertLess(np.abs(pct_change_sigma), .01)

        # Check that the two models include the same set of coefficients.
        raw_coef = raw_summary.coefficients
        suf_coef = suf_summary.coefficients

        raw_included = raw_coef["inc_prob"] > .5
        suf_included = suf_coef["inc_prob"] > .5
        self.assertEqual(
            set(raw_coef[raw_included].index),
            set(suf_coef[suf_included].index)
        )

    def test_plots(self):
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
        ax1 = model.plot_inclusion(inclusion_threshold=.1)
        ax2 = model.plot_coefficients(inclusion_threshold=.1)
        self.assertIsInstance(ax1, plt.Axes)
        self.assertIsInstance(ax2, plt.Axes)

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


_debug_mode = False

if _debug_mode:
    import pdb  # noqa

    # Turn warnings into errors.
    import warnings
    warnings.simplefilter("error")

    # Run the test you are trying to debug here.  Instantiate the test class,
    # then call the problematic test.  Call pdb.pm() in the event of an
    # exception.
    print("Hello, world!")

    rig = SpikeSlabTest()
    if hasattr(rig, "setUpClass"):
        rig.setUpClass()
    if hasattr(rig, "setUp"):
        rig.setUp()

    rig.test_plots()

    print("Goodbye, cruel world!")

if __name__ == "__main__":
    unittest.main()
