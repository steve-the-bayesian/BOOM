import unittest
import BayesBoom.boom as boom
import numpy as np


class MvnModelTest(unittest.TestCase):
    def setUp(self):
        np.random.seed(8675309)
        self.data = np.random.randn(100, 3)
        self.Sigma = boom.SpdMatrix(np.array(
            [[1, .8, -.3],
             [.8, 2, -.6],
             [-.3, -.6, 4]]))

        chol = boom.Cholesky(self.Sigma)
        L = chol.getL(True).to_numpy()
        self.data = self.data @ L.T
        self.mu = np.array([1, 2, -3.0])
        self.data += self.mu
        self.mu = boom.Vector(self.mu)

    def test_moments(self):
        model = boom.MvnModel(self.mu, self.Sigma)
        self.assertLess((model.mu - self.mu).normsq(), 1e-5)
        self.assertTrue(np.allclose(model.Sigma.to_numpy(),
                                    self.Sigma.to_numpy()))

    def test_data(self):
        model = boom.MvnModel(self.mu, self.Sigma)
        model.set_data(boom.Matrix(self.data))
        model.mle()
        self.assertLess(model.siginv.Mdist(self.mu, model.mu), .05)

    def test_parameters(self):
        """When parameters are modified outside the object, the object properties
         should change.  This is testing that pointers are being stored.
         """
        zeros = boom.Vector(np.array([0.0, 0.0, 0.0]))
        model = boom.MvnModel(zeros, self.Sigma)
        mu_prm = model.mean_parameter
        new_mu = boom.Vector(np.array([3.0, 2.0, 1.0]))
        mu_prm.set(new_mu)
        self.assertLess((model.mu - new_mu).normsq(), 1e-5)


if __name__ == "__main__":
    unittest.main()
