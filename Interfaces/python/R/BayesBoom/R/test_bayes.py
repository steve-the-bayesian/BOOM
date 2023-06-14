import unittest
import BayesBoom.R as R
import numpy as np


class TestRegSuf(unittest.TestCase):

    def setUp(self):
        np.random.seed(8675309)
        sample_size = 1000
        xdim = 4
        self._x = np.random.randn(sample_size, xdim)
        self._x[:, 0] = 1.0
        self._y = np.random.randn(sample_size)

    def test_eval(self):
        xtx = self._x.T @ self._x
        xty = self._x.T @ self._y
        sdy = np.std(self._y, ddof=1)
        sample_size = self._y.size
        ybar = self._y.mean()
        xbar = self._x.mean(axis=0)
        suf = R.RegSuf(xtx, xty, sdy, sample_size, ybar, xbar)
        yty = np.sum(self._y ** 2)

        bs = suf.boom()
        self.assertTrue(np.allclose(bs.xtx.to_numpy(), xtx))
        self.assertTrue(np.allclose(bs.xty.to_numpy(), xty))
        self.assertEqual(bs.sample_size, sample_size)
        self.assertAlmostEqual(bs.yty, yty, 4)


_debug_mode = False

if _debug_mode:
    import pdb  # noqa

    # Turn warnings into errors.
    # warnings.simplefilter("error")

    # Run the test you are trying to debug here.  Instantiate the test class,
    # then call the problematic test.  Call pdb.pm() in the event of an
    # exception.
    print("Hello, world!")

    rig = TestRegSuf()
    if hasattr(rig, "setUpClass"):
        rig.setUpClass()
    if hasattr(rig, "setUp"):
        rig.setUp()

    rig.test_eval()

    print("Goodbye, cruel world!")

else:
    if __name__ == "__main__":
        unittest.main(verbosity=2)
