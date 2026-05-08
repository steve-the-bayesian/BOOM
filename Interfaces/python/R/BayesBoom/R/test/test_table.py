import unittest
import BayesBoom.R as R
import numpy as np
import pandas as pd

stooges = ["Larry", "Moe", "Curly", "Shemp"]
colors = ["Red", "Blue", "Green"]
states = ["California", "Texas", "Massachussetts"]


class TestTable(unittest.TestCase):

    def setUp(self):
        np.random.seed(8675309)

    def test_single_numpy(self):
        x = np.random.choice(stooges, 100)
        tab = R.table(x)
        self.assertIsInstance(tab, pd.Series)

    def test_two_numpy(self):
        sample_size = 100
        x = np.random.choice(stooges, sample_size)
        y = np.random.choice(colors, sample_size)
        tab = R.table(x, y)
        self.assertIsInstance(tab, pd.DataFrame)

    def test_three_numpy(self):
        sample_size = 100
        x = np.random.choice(stooges, sample_size)
        y = np.random.choice(colors, sample_size)
        z = np.random.choice(states, sample_size)
        tab = R.table(x, y, z)
        self.assertIsInstance(tab, pd.DataFrame)

        # Right now the "table" we get out of a 3-factor crosstab is a
        # pd.DataFrame.  It really should be a multi-way table.


_debug_mode = False

if _debug_mode:
    import pdb  # noqa

    # Turn warnings into errors.
    # warnings.simplefilter("error")

    # Run the test you are trying to debug here.  Instantiate the test class,
    # then call the problematic test.  Call pdb.pm() in the event of an
    # exception.
    print("Hello, world!")

    rig = TestTable()
    if hasattr(rig, "setUpClass"):
        rig.setUpClass()
    if hasattr(rig, "setUp"):
        rig.setUp()

    rig.test_three_numpy()

    print("Goodbye, cruel world!")

else:
    if __name__ == "__main__":
        unittest.main(verbosity=2)
