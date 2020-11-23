import unittest
from BayesBoom.spikeslab import dot, lm_spike
import numpy as np
import pandas as pd
import scipy.sparse


class SpikeSlabTest(unittest.TestCase):
    def setUp(self):
        np.random.seed(8675309)

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


if __name__ == "__main__":
    unittest.main()
