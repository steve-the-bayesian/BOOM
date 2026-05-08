import unittest
import BayesBoom.R as R
import numpy as np


class TestBase(unittest.TestCase):

    def setUp(self):
        np.random.seed(8675309)

    def test_print_time_interval(self):
        self.assertEqual(R.print_time_interval(3), "3.000 seconds")


_debug_mode = False

if _debug_mode:
    import pdb  # noqa

    # Turn warnings into errors.
    # warnings.simplefilter("error")

    # Run the test you are trying to debug here.  Instantiate the test class,
    # then call the problematic test.  Call pdb.pm() in the event of an
    # exception.
    print("Hello, world!")

    rig = TestBase()
    if hasattr(rig, "setUpClass"):
        rig.setUpClass()
    if hasattr(rig, "setUp"):
        rig.setUp()

    rig.test_print_time_interval()

    print("Goodbye, cruel world!")

else:
    if __name__ == "__main__":
        unittest.main(verbosity=2)
