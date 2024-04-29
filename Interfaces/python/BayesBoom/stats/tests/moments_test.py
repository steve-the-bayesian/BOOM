import unittest
import BayesBoom.boom as boom
import numpy as np
from BayesBoom.boom import mean, var, cor


class MomentsTest(unittest.TestCase):

    def setUp(self):
        np.random.seed(8675309)

    def test_vector(self):
        y = np.random.randn(10000) * 1.7 - 8.2
        y = boom.Vector(y)
        m = boom.mean(y)
        self.assertLess(abs(-8.2 - m), .025)

    def test_matrix(self):
        y = np.random.randn(10000, 3)
        Sigma = boom.SpdMatrix(np.array(
            [[1, .8, -.6],
             [.8, 2, -.8],
             [-.6, -.8, 4]]))
        chol = boom.Cholesky(Sigma)
        R = chol.getLT()
        y = y @ R.to_numpy()
        mu = np.array([1, 2, -3], dtype=float)
        y = y + mu
        mu = boom.Vector(mu)
        y = boom.Matrix(y)
        meany = mean(y)
        self.assertLess((meany - mu).normsq(), .01)

        V = var(y)
        self.assertLess((V.diag() - Sigma.diag()).normsq(), .05)

        R = cor(y)
        Rtrue = Sigma.to_numpy()
        for i in range(3):
            for j in range(3):
                Rtrue[i, j] = Sigma[i, j] / np.sqrt(Sigma[i, i] * Sigma[j, j])
        Rtrue = boom.SpdMatrix(Rtrue)

        self.assertLess((Rtrue - R).max_abs(), .01)


if __name__ == "__main__":
    unittest.main()
