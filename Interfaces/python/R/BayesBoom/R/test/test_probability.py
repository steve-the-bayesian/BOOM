import unittest
import BayesBoom.R as R
from BayesBoom.R import dmvn, rmvn
import numpy as np
import scipy.stats as ss


class TestDmvn(unittest.TestCase):

    def setUp(self):
        np.random.seed(8675309)

    def test_eval(self):
        Z = np.random.randn(10, 3)
        Sigma = Z.T @ Z
        Sigma_chol = np.linalg.cholesky(Sigma)

        mu = np.random.randn(3)

        y = np.random.randn(5, 3)
        for i in range(5):
            y[i, :] = Sigma_chol @ y[i, :] + mu

        prob = dmvn(y, mu, Sigma)
        obj = ss.multivariate_normal(mean=mu, cov=Sigma)
        self.assertTrue(np.allclose(prob, obj.pdf(y)))

        # Test logscale
        logprob = dmvn(y, mu, Sigma, logscale=True)
        self.assertTrue(np.allclose(logprob, obj.logpdf(y)))

        # Test with parameter arrays.
        Z = np.random.randn(10, 3)
        Sigma2 = Z.T @ Z

        Z = np.random.randn(10, 3)
        Sigma3 = Z.T @ Z
        Sigma_array = np.array([Sigma, Sigma2, Sigma3])

        prob = dmvn(y[:3, :], mu, Sigma_array)
        self.assertAlmostEqual(prob[0], dmvn(y[0, :], mu, Sigma_array[0, :, :]))
        self.assertAlmostEqual(prob[1], dmvn(y[1, :], mu, Sigma_array[1, :, :]))
        self.assertAlmostEqual(prob[2], dmvn(y[2, :], mu, Sigma_array[2, :, :]))

        mu_array = np.random.randn(4, 3)
        prob = dmvn(y[:4, :], mu_array, Sigma)
        self.assertAlmostEqual(prob[0], dmvn(y[0, :], mu_array[0, :], Sigma))
        self.assertAlmostEqual(prob[1], dmvn(y[1, :], mu_array[1, :], Sigma))
        self.assertAlmostEqual(prob[2], dmvn(y[2, :], mu_array[2, :], Sigma))
        self.assertAlmostEqual(prob[3], dmvn(y[3, :], mu_array[3, :], Sigma))


class TestRmvn(unittest.TestCase):
    def setUp(self):
        np.random.seed(8675309)

    def test_draws(self):
        mu = np.array([1, 3, 7])
        draws = rmvn(10, mu, np.diag(np.ones(3)))
        self.assertEqual(draws.shape, (10, 3))

        draws = rmvn(10000, mu, np.diag(np.ones(3)))
        epsilon = 5 * 1.0 / np.sqrt(draws.shape[0])
        self.assertTrue(np.all(np.abs(draws.mean(axis=0) - mu) < epsilon))

        Sigma = np.array([
            [1.0, 0.4, -.3],
            [0.4, 2.0, 0.9],
            [-.3, 0.9, 8.0]])

        draws = rmvn(100000, mu, Sigma)
        epsilon = 5 * np.sqrt(np.diag(Sigma) / draws.shape[0])
        self.assertTrue(np.all(np.abs(draws.mean(axis=0) - mu) < epsilon))

        V = R.var(draws)
        self.assertTrue(np.all(np.abs(V - Sigma) < .05))

    def test_multiple_sigma_values(self):

        Sigma1 = np.array([
            [1.0, 0.4, -.3],
            [0.4, 2.0, 0.9],
            [-.3, 0.9, 8.0]])
        Sigma2 = 9 * Sigma1

        mu1 = np.array([1, 3, 7])
        mu2 = np.array([70, -30, 1])

        n1 = 1000
        n2 = 2000
        mu = np.array([mu1] * n1 + [mu2] * n2)
        Sigma = np.array([Sigma1] * n1 + [Sigma2] * n2)

        draws = R.rmvn(n1 + n2, mu, Sigma)
        self.assertEqual(draws.shape, (3000, 3))
        epsilon1 = 5 * np.sqrt(np.diag(Sigma1) / n1)
        self.assertTrue(np.all((draws[:n1, :].mean(axis=0) - mu1) < epsilon1))

        V1 = R.var(draws[:n1, :])
        relative_error = np.max(np.abs((V1 - Sigma1) / Sigma1))
        self.assertLess(relative_error, .2)

        epsilon2 = 5 * np.sqrt(np.diag(Sigma2) / n2)
        self.assertTrue(np.all((draws[n1:, :].mean(axis=0) - mu2) < epsilon2))
        V2 = R.var(draws[n1:, :])
        relative_error = np.max(np.abs((V2 - Sigma2) / Sigma2))
        self.assertLess(relative_error, .2)


_debug_mode = False

if _debug_mode:
    import pdb  # noqa

    # Turn warnings into errors.
    # warnings.simplefilter("error")

    # Run the test you are trying to debug here.  Instantiate the test class,
    # then call the problematic test.  Call pdb.pm() in the event of an
    # exception.
    print("Hello, world!")

    rig = TestRmvn()
    if hasattr(rig, "setUpClass"):
        rig.setUpClass()
    if hasattr(rig, "setUp"):
        rig.setUp()

    rig.test_multiple_sigma_values()

    print("Goodbye, cruel world!")

else:
    if __name__ == "__main__":
        unittest.main(verbosity=2)
