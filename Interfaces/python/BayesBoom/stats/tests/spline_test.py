import unittest
import BayesBoom.boom as boom
import numpy as np


class SplineTest(unittest.TestCase):

    def setUp(self):
        self.knots = [1, 2, 3]

    def test_bspline(self):
        spline = boom.Bspline(boom.Vector(self.knots))
        scalar = 2.1
        basis = spline.basis(scalar)
        self.assertTrue(isinstance(basis, boom.Vector))

        self.assertEqual(spline.degree, 3)
        self.assertEqual(spline.order, 4)
        self.assertTrue(isinstance(spline.knots(), boom.Vector))
        self.assertEqual(spline.knots().size, 3)
        self.assertTrue(np.allclose(spline.knots().to_numpy(),
                                    np.array([1.0, 2.0, 3.0])))
        self.assertEqual(spline.dim, 3 - 1 + 3)
        self.assertEqual(spline.dim, basis.size)

        expected = np.array([0, 0.18224999999999994, 0.48599999999999999,
                             0.3307500000000001, 0.0010000000000000026])
        self.assertTrue(np.allclose(basis.to_numpy(), expected))

        basis_matrix = spline.basis_matrix(np.array([2.1, 2.1]))
        ra = basis_matrix.to_numpy()
        self.assertTrue(np.allclose(ra[0, :], expected))
        self.assertTrue(np.allclose(ra[1, :], expected))

    @staticmethod
    def test_again():
        knots = np.arange(-3, 4)
        spline = boom.Bspline(boom.Vector(knots))
        spline.basis(1.2)

if __name__ == "__main__":
    unittest.main()
