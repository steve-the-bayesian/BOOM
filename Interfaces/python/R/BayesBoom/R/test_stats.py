import unittest
import BayesBoom.R as R
import numpy as np
import pandas as pd
import scipy.sparse


class TestMoments(unittest.TestCase):

    def setUp(self):
        np.random.seed(8675309)

    def test_eval(self):
        matrix = np.random.randn(10, 3)
        frame = pd.DataFrame(matrix, columns=["Larry", "Moe", "Curly"])
        sparse = scipy.sparse.csr_matrix(matrix)

        self.assertTrue(isinstance(R.mean(frame), float))
        self.assertTrue(isinstance(R.mean(frame, axis=0), pd.Series))

        self.assertTrue(np.all(R.mean(frame, axis=0).index == frame.columns))

        self.assertTrue(np.allclose(R.mean(matrix, axis=0),
                                    R.mean(frame, axis=0)))
        self.assertTrue(np.allclose(R.mean(matrix, axis=0),
                                    R.mean(sparse, axis=0)))
        self.assertTrue(np.allclose(R.mean(matrix, axis=0),
                                    np.mean(matrix, axis=0)))

        self.assertAlmostEqual(
            R.mean(matrix[:, 0]),
            R.mean(frame.iloc[:, 0]))
        self.assertAlmostEqual(
            R.mean(matrix[:, 0]),
            R.mean(sparse[:, 0]))
        self.assertAlmostEqual(
            R.mean(matrix[:, 0]),
            np.mean(matrix[:, 0]))

        self.assertTrue(isinstance(R.sd(frame), pd.Series))
        self.assertTrue(np.all(R.sd(frame).index == frame.columns))

        self.assertTrue(np.allclose(R.sd(matrix, axis=0),
                                    R.sd(frame)))
        self.assertTrue(np.allclose(R.sd(matrix, axis=0),
                                    R.sd(sparse, axis=0)))
        self.assertTrue(np.allclose(R.sd(matrix, axis=0),
                                    np.std(matrix, axis=0, ddof=1)))

        self.assertAlmostEqual(
            R.sd(matrix[:, 0]),
            R.sd(frame.iloc[:, 0]))
        self.assertAlmostEqual(
            R.sd(matrix[:, 0]),
            R.sd(sparse[:, 0]))
        self.assertAlmostEqual(
            R.sd(matrix[:, 0]),
            np.std(matrix[:, 0], ddof=1))

        x = np.matrix([])
        self.assertAlmostEqual(0.0, R.sd(x))

        raw = np.random.randn(10)
        x = np.matrix(raw)
        self.assertAlmostEqual(R.sd(x), R.sd(raw))


_debug_mode = False

if _debug_mode:
    import pdb  # noqa

    # Turn warnings into errors.
    # warnings.simplefilter("error")

    # Run the test you are trying to debug here.  Instantiate the test class,
    # then call the problematic test.  Call pdb.pm() in the event of an
    # exception.
    print("Hello, world!")

    rig = TestMoments()
    if hasattr(rig, "setUpClass"):
        rig.setUpClass()
    if hasattr(rig, "setUp"):
        rig.setUp()

    rig.test_eval()

    print("Goodbye, cruel world!")

else:
    if __name__ == "__main__":
        unittest.main(verbosity=2)
