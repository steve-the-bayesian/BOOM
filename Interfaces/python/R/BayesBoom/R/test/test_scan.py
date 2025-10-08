import unittest
import BayesBoom.R as R
import numpy as np


class TestScan(unittest.TestCase):

    def setUp(self):
        np.random.seed(8675309)
        self._x = np.array([1.0, 2, 3, 4])
        self._fname = "blah.txt"

    def test_base_case(self):
        np.savetxt(self._fname, self._x)
        x = R.scan(self._fname)
        self.assertTrue(np.allclose(x, self._x))

    def tearDown(self):
        R.delete_if_present(self._fname)
        
_debug_mode = False

if _debug_mode:
    import pdb  # noqa

    # Turn warnings into errors.
    # warnings.simplefilter("error")

    # Run the test you are trying to debug here.  Instantiate the test class,
    # then call the problematic test.  Call pdb.pm() in the event of an
    # exception.
    print("Hello, world!")

    rig = TestScan()
    if hasattr(rig, "setUpClass"):
        rig.setUpClass()
    if hasattr(rig, "setUp"):
        rig.setUp()

    rig.test_base_case()

    if hasattr(rig, "tearDown"):
        rig.tearDown()

    print("Goodbye, cruel world!")

else:
    if __name__ == "__main__":
        unittest.main(verbosity=2)
