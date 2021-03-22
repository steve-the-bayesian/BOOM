import unittest
import BayesBoom.R as R
import BayesBoom.boom as boom
import numpy as np
import pandas as pd


class TestUtilities(unittest.TestCase):

    def setUp(self):
        pass

    def test_conversions(self):
        x = [1, 2, 3]
        v = R.to_boom_vector(x)
        self.assertIsInstance(v, boom.Vector)

        x = pd.Series(x, dtype="int")
        v = R.to_boom_vector(x)
        self.assertIsInstance(v, boom.Vector)

    def test_numerics(self):
        numeric_df = pd.DataFrame(np.random.randn(10, 3))
        self.assertTrue(R.is_all_numeric(numeric_df))

        non_numeric = numeric_df.copy()
        non_numeric["text"] = "foo"
        self.assertFalse(R.is_all_numeric(non_numeric))


_debug_mode = True

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
