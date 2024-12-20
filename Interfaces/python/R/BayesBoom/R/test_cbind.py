import unittest
import numpy as np
import pandas as pd

from BayesBoom.R import cbind


class TestCbind(unittest.TestCase):

    def setUp(self):
        np.random.seed(8675309)

    def test_cbind_numpy(self):
        x = np.random.randn(3)
        ans = cbind(1, x)
        self.assertEqual(ans.shape, (3, 2))
        self.assertTrue(np.allclose(ans[:, 1], x))
        self.assertTrue(np.allclose(ans[:, 0], np.full(3, 1.0)))

        X = np.random.randn(3, 4)
        ans = cbind(x, X, 1.0)
        self.assertEqual(ans.shape, (3, 6))
        self.assertTrue(np.allclose(ans[:, 0], x))
        self.assertTrue(np.allclose(ans[:, 1], X[:, 0]))
        self.assertTrue(np.allclose(ans[:, 2], X[:, 1]))
        self.assertTrue(np.allclose(ans[:, 5], np.ones(3)))

        ans = cbind(x, x)
        self.assertEqual(ans.shape, (3, 2))
        self.assertTrue(np.allclose(ans[:, 0], x))
        self.assertTrue(np.allclose(ans[:, 1], x))

    def test_cbind_pandas(self):
        x = np.random.randn(3)
        y = pd.Series(np.random.randn(3))
        z = 2.0
        w = np.random.randn(3, 4)
        d = pd.DataFrame(np.random.randn(3, 2), columns=["Fred", "Barney"])
        foo = cbind(x, y)
        self.assertTrue(isinstance(foo, pd.DataFrame))
        self.assertTrue(foo.shape, (3, 2))
        self.assertTrue(np.allclose(foo.iloc[:, 0], x))
        self.assertTrue(np.allclose(foo.iloc[:, 1], y))
        self.assertEqual(foo.columns[0], "x")
        self.assertEqual(foo.columns[1], "y")

        bar = cbind(x, y, z)
        self.assertEqual(bar.shape, (3, 3))
        self.assertTrue(np.allclose(bar.iloc[:, 2], np.full(3, 2.0)))
        self.assertEqual(bar.columns[0], "x")
        self.assertEqual(bar.columns[1], "y")
        self.assertEqual(bar.columns[2], "z")

        baz = cbind(x, y, z, w)
        self.assertEqual(baz.shape, (3, 7))
        self.assertTrue(np.allclose(baz.iloc[:, 3], w[:, 0]))
        self.assertEqual(baz.columns[0], "x")
        self.assertEqual(baz.columns[1], "y")
        self.assertEqual(baz.columns[2], "z")
        self.assertEqual(baz.columns[3], "w0")
        self.assertEqual(baz.columns[4], "w1")
        self.assertEqual(baz.columns[5], "w2")
        self.assertEqual(baz.columns[6], "w3")

        qux = cbind(x, y, z, w, d)
        self.assertEqual(qux.shape, (3, 9))
        self.assertTrue(np.allclose(qux.iloc[:, 7], d.iloc[:, 0]))
        self.assertEqual(qux.columns[0], "x")
        self.assertEqual(qux.columns[1], "y")
        self.assertEqual(qux.columns[2], "z")
        self.assertEqual(qux.columns[3], "w0")
        self.assertEqual(qux.columns[4], "w1")
        self.assertEqual(qux.columns[5], "w2")
        self.assertEqual(qux.columns[6], "w3")
        self.assertEqual(qux.columns[7], "Fred")
        self.assertEqual(qux.columns[8], "Barney")

        blah = cbind(d, 1.8)
        self.assertEqual(blah.shape, (3, 3))
        self.assertEqual(blah.columns[0], "Fred")
        self.assertEqual(blah.columns[1], "Barney")
        self.assertEqual(blah.columns[2], "1.8")


_debug_mode = True

if _debug_mode:
    import pdb  # noqa

    # Turn warnings into errors.
    # warnings.simplefilter("error")

    # Run the test you are trying to debug here.  Instantiate the test class,
    # then call the problematic test.  Call pdb.pm() in the event of an
    # exception.
    print("Hello, world!")

    rig = TestCbind()
    if hasattr(rig, "setUpClass"):
        rig.setUpClass()
    if hasattr(rig, "setUp"):
        rig.setUp()

    rig.test_cbind_numpy()
    rig.test_cbind_pandas()

    print("Goodbye, cruel world!")

else:
    if __name__ == "__main__":
        unittest.main(verbosity=2)
