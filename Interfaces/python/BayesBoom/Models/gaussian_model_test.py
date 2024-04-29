import unittest
import BayesBoom.boom as boom
import numpy as np


class GaussianModelTest(unittest.TestCase):

    def setUp(self):
        self.data = np.array([1, 2, 3])
        np.random.seed(8675309)

    def test_moments(self):
        model = boom.GaussianModel(1, 2)
        self.assertEqual(1.0, model.mean)
        self.assertEqual(2.0, model.sd)
        self.assertEqual(4.0, model.variance)

    def test_data(self):
        model = boom.GaussianModel(0, 1)
        mu = -16
        sigma = 7
        data = np.random.randn(10000) * sigma + mu
        model.set_data(boom.Vector(data))
        model.mle()
        self.assertLess(np.abs(model.mean - mu),
                        4 * sigma / np.sqrt(len(data)))
        self.assertLess(np.abs(model.sd - sigma), .1)

    def test_parameters(self):
        """When parameters are modified outside the object, the object properties
        should change.  This is testing that pointers are being stored.

        """
        model = boom.GaussianModel(0, 1)
        mu_prm = model.mean_parameter
        mu_prm.set(2.0)
        self.assertAlmostEqual(model.mean, 2.0)

    @staticmethod
    def test_mcmc():
        model = boom.GaussianModel()
        mu = -16
        sigma = 7
        data = np.random.randn(10000) * sigma + mu
        model.set_data(boom.Vector(data))

        mean_prior = boom.GaussianModelGivenSigma(
            model.sigsq_parameter,
            mu,
            1.0)
        sigsq_prior = boom.ChisqModel(1.0, sigma)
        sampler = boom.GaussianConjugateSampler(
            model,
            mean_prior,
            sigsq_prior)
        model.set_method(sampler)
        for _ in range(100):
            model.sample_posterior()


class ZeroMeanGaussianModelTest(unittest.TestCase):

    def setUp(self):
        self.model = boom.ZeroMeanGaussianModel(2.3)

    def test_parameters(self):
        self.assertAlmostEqual(self.model.sd, 2.3)
        self.assertEqual(0.0, self.model.mean)

    def test_mcmc(self):
        true_sigma = 2.3
        data = np.random.randn(100) * true_sigma
        prior = boom.ChisqModel(1.0, 1.0)
        self.model.set_data(boom.Vector(data))
        sampler = boom.ZeroMeanGaussianConjSampler(self.model, prior)
        self.model.set_method(sampler)
        niter = 1000
        draws = np.zeros(niter)
        for i in range(niter):
            self.model.sample_posterior()
            draws[i] = self.model.sigma
        self.assertNotAlmostEqual(draws[0], draws[-1])


if __name__ == "__main__":
    unittest.main()
