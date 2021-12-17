import unittest
# from BayesBoom.R import dmvn
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

        mu_array = np.random.randn(4,3)
        prob = dmvn(y[:4, :], mu_array, Sigma)
        self.assertAlmostEqual(prob[0], dmvn(y[0, :], mu_array[0, :], Sigma))
        self.assertAlmostEqual(prob[1], dmvn(y[1, :], mu_array[1, :], Sigma))
        self.assertAlmostEqual(prob[2], dmvn(y[2, :], mu_array[2, :], Sigma))
        self.assertAlmostEqual(prob[3], dmvn(y[3, :], mu_array[3, :], Sigma))


_debug_mode = True

if _debug_mode:
    import pdb  # noqa

    # Turn warnings into errors.
    # warnings.simplefilter("error")

    # Run the test you are trying to debug here.  Instantiate the test class,
    # then call the problematic test.  Call pdb.pm() in the event of an
    # exception.
    print("Hello, world!")

    rig = TestDmvn()
    if hasattr(rig, "setUpClass"):
        rig.setUpClass()
    if hasattr(rig, "setUp"):
        rig.setUp()

    rig.test_eval()

    print("Goodbye, cruel world!")

else:
    if __name__ == "__main__":
        unittest.main(verbosity=2)
