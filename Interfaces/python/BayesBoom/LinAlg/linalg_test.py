import unittest
import BayesBoom as boom
import numpy as np


class LinAlgTest(unittest.TestCase):

    def setUp(self):
        np.random.seed(8675309)

    def test_vector(self):
        v = boom.Vector(3, -2.8)
        self.assertEqual(v.size, 3)
        vn = v.to_numpy()
        self.assertTrue(np.array_equal(
            vn, np.array([-2.8, -2.8, -2.8])))

        v1 = boom.Vector(np.array([1.0, 2, 3]))
        v2 = boom.Vector(np.array([3.0, 2, 1]))
        v3 = v1 / v2
        self.assertTrue(np.array_equal(
            v3.to_numpy(),
            np.array([1.0 / 3, 1.0, 3.0])))
        self.assertEqual(len(v), v.length)

    def test_matrix(self):
        m = boom.Matrix(np.array([[1.0, 2], [3, 4.0]]))
        mn = m.to_numpy()
        m2 = m * m
        m2n = mn @ mn
        self.assertTrue(np.array_equal(m2.to_numpy(), m2n))

        # Test assignment
        m[0, 1] = 8.3
        self.assertEqual(m[0, 1], 8.3)
        self.assertEqual(m[0, 0], 1.0)

    def test_spd(self):
        X = np.random.randn(100, 4)
        xtx = X.T @ X / X.shape[0]
        S = boom.SpdMatrix(xtx)

    def test_vector_view(self):
        v = boom.Vector(np.array([1.0, 2.0, -3.0]))
        vv = boom.VectorView(v)
        vv[0] = -0.1
        self.assertEqual(v[0], vv[0])
        vv /= 2.0
        self.assertEqual(v[1], 1.0)


if __name__ == "__main__":
    unittest.main()
