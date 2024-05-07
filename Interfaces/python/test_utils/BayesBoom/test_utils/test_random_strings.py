import unittest
from BayesBoom.test_utils import random_strings
import numpy as np


class TestRandomStrings(unittest.TestCase):
    def setUp(self):
        np.random.seed(8675309)

    def test_length(self):
        strings = random_strings(100, 4)
        self.assertEqual(100, len(strings))
        self.assertEqual(4, len(strings[0]))

    def test_uniqueness(self):
        non_unique = random_strings(1000, 2, ensure_unique=False)
        self.assertEqual(1000, len(non_unique))

        
_debug_mode = True

if _debug_mode:
    import pdb  # noqa

    # Turn warnings into errors.
    # import warnings
    # warnings.simplefilter("error")

    # Run the test you are trying to debug here.  Instantiate the test class,
    # then call the problematic test.  Call pdb.pm() in the event of an
    # exception.
    print("Hello, world!")

    rig = TestRandomStrings()
    if hasattr(rig, "setUpClass"):
        rig.setUpClass()
    if hasattr(rig, "setUp"):
        rig.setUp()

    rig.test_length()
    rig.test_uniqueness()

    print("Goodbye, cruel world!")

else:
    if __name__ == "__main__":
        unittest.main()
        
