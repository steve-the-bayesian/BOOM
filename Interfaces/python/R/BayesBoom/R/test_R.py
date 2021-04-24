import unittest
import BayesBoom.R as R
import numpy as np
import pandas as pd


class TestRfuns(unittest.TestCase):

    def setUp(self):
        np.random.seed(8675309)

    def test_paste(self):
        foo = R.paste("X", [1, 2, 3])
        self.assertEqual(foo, ["X 1", "X 2", "X 3"])

        bar = R.paste("X", [1, 2, 3], sep="")
        self.assertEqual(bar, ["X1", "X2", "X3"])

        baz = R.paste([1, 2, "X"], [4, 5, 6])
        self.assertEqual(baz, ["1 4", "2 5", "X 6"])

        foo = R.paste("X", pd.Series([1, 2, 3]), sep="")
        self.assertEqual(foo, ["X1", "X2", "X3"])

        f = R.paste("X", [1, 2, 3], sep="", collapse=" ")
        self.assertEqual(f, "X1 X2 X3")


_debug_mode = False

if _debug_mode:
    import pdb  # noqa

    # Turn warnings into errors.
    # warnings.simplefilter("error")

    # Run the test you are trying to debug here.  Instantiate the test class,
    # then call the problematic test.  Call pdb.pm() in the event of an
    # exception.
    print("Hello, world!")

    rig = TestUtilities()
    if hasattr(rig, "setUpClass"):
        rig.setUpClass()
    if hasattr(rig, "setUp"):
        rig.setUp()

    rig.test_conversions()
    rig.test_numerics()

    print("Goodbye, cruel world!")

else:
    if __name__ == "__main__":
        unittest.main(verbosity=2)
