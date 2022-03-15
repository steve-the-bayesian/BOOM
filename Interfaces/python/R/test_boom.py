import unittest
import numpy as np
import BayesBoom.boom as boom


class TestBoom(unittest.TestCase):

    def setUp(self):
        pass

    def test_rng(self):
        # As long an object holds the GlobalRng in state in python, seeding
        # works as normal.
        rng = boom.GlobalRng.rng

        rng1 = boom.GlobalRng.rng
        rng.seed(123)
        x1 = np.array([rng1() for x in range(10)])

        rng2 = boom.GlobalRng.rng
        rng.seed(123)
        x2 = np.array([rng2() for x in range(10)])
        np.testing.assert_array_almost_equal(x1, x2)




_debug_mode = True

if _debug_mode:
    import pdb  # noqa

    # Turn warnings into errors.
    # warnings.simplefilter("error")

    # Run the test you are trying to debug here.  Instantiate the test class,
    # then call the problematic test.  Call pdb.pm() in the event of an
    # exception.
    print("Hello, world!")

    rig = TestBoom()

    if hasattr(rig, "setUpClass"):
        rig.setUpClass()
    if hasattr(rig, "setUp"):
        rig.setUp()

    rig.test_rng()

    print("Goodbye, cruel world!")

else:
    if __name__ == "__main__":
        unittest.main(verbosity=2)
