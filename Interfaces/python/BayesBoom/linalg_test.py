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

    def test_matrix(self):
        m = boom.Matrix(np.array([[1.0, 2], [3, 4.0]]))
        mn = m.to_numpy()
        m2 = m * m
        m2n = mn @ mn
        self.assertTrue(np.array_equal(m2.to_numpy(), m2n))


if __name__ == "__main__":
    unittest.main()
