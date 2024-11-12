import unittest
import BayesBoom.boom as boom
import BayesBoom.R as R
import numpy as np

class LinAlgTest(unittest.TestCase):

    def setUp(self):
        np.random.seed(8675309)

    def test_vector(self):
        v = boom.Vector(np.full(3, -2.8))
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

    # Check that a numpy array can be implicitly converted to a Matrix.  This
    # is done by adding a Matrix to an np.array.  The result is a Matrix.
    def test_implicit_conversion(self):
        X = np.random.randn(3, 4)
        bX = boom.Matrix(X)
        Y = np.random.randn(3, 4)
        Z = X + Y
        bZ = bX + Y
        self.assertTrue(isinstance(bZ, boom.Matrix))
        # Check that numpy addition and boom addition get the same answer.
        # This also checks fortran vs C ordering.
        delta = bZ - Z
        self.assertLess(delta.max_abs(), 1e-15)

    def test_spd(self):
        X = np.random.randn(100, 4)
        xtx = X.T @ X / X.shape[0]
        S = boom.SpdMatrix(xtx)
        self.assertEqual(S.nrow, 4)
        self.assertEqual(S.ncol, 4)
        self.assertTrue(np.allclose(S.to_numpy(), xtx))

    def test_vector_view(self):
        v = boom.Vector(np.array([1.0, 2.0, -3.0]))
        vv = boom.VectorView(v)
        vv[0] = -0.1
        self.assertEqual(v[0], vv[0])
        vv /= 2.0
        self.assertEqual(v[1], 1.0)

    def test_labelled_matrix(self):
        raw_data = np.random.randn(3, 4)
        row_names = ["Larry", "Moe", "Curly"]
        col_names = ["Doe", "Ray", "Me", "Fa"]
        m = boom.LabelledMatrix(boom.Matrix(raw_data), row_names, col_names)

        self.assertEqual(row_names, m.row_names)
        self.assertEqual(col_names, m.col_names)
        self.assertTrue(np.allclose(raw_data, m.to_numpy()))

        m2 = boom.LabelledMatrix(boom.Matrix(raw_data), row_names, [])
        self.assertEqual(row_names, m2.row_names)
        self.assertEqual([], m2.col_names)
        self.assertTrue(np.allclose(raw_data, m2.to_numpy()))

        m3 = boom.LabelledMatrix(boom.Matrix(raw_data), [], col_names)
        self.assertEqual([], m3.row_names)
        self.assertEqual(col_names, m3.col_names)
        self.assertTrue(np.allclose(raw_data, m3.to_numpy()))

    def test_array(self):
        # x = np.random.randn(2,3,4)
        x = np.array(
            [[[ 0.33726119, -1.16232657, -0.32731355,  0.21659849],
              [-0.48883675, -0.58757439,  0.18716781, -0.12464536],
              [-0.53821583,  0.58069231,  0.19968848,  0.6311756 ]],
             [[ 0.88622023, -0.6579181 ,  1.01357032,  0.64801707],
              [ 1.58231319,  0.29383011,  0.58968301, -0.36975214],
              [ 1.38582045,  1.75563478,  0.57981513, -0.47530515]]])

        XX = R.to_boom_array(x)

        # Check the mapping from numpy to BOOM.
        for i in range(2):
            for j in range(3):
                for k in range(4):
                    self.assertAlmostEqual(x[i, j, k], XX[i, j, k])

        # Check the reverse mapping from BOOM to numppy.
        xxx = XX.to_numpy()
        self.assertTrue(np.allclose(x, xxx))

        # for i in range(2):
        #     for j in range(3):
        #         for k in range(4):
        #             self.assertAlmostEqual(x[i, j, k], xxx[i, j, k])

        ind = boom.argmax_random_tie(XX, apply_over=[2]);
        self.assertEqual(ind.ndim, 2)
        self.assertEqual(ind.dim(0), 2)
        self.assertEqual(ind.dim(1), 3)

        correct_argmax = np.array(
            [[0, 2, 3],
             [2, 0, 1]], dtype=int)

        python_ind = ind.to_numpy()
        python_ind_int = python_ind.astype(int)
        for i in range(2):
            for j in range(3):
                self.assertAlmostEqual(ind[i, j], correct_argmax[i, j])
                self.assertAlmostEqual(python_ind[i, j], correct_argmax[i, j]);
                self.assertAlmostEqual(python_ind_int[i, j], correct_argmax[i, j]);



_debug_mode = False

if _debug_mode:
    import pdb
    rig = LinAlgTest()
    rig.setUp()
    rig.test_array()

elif __name__ == "__main__":
    unittest.main()
