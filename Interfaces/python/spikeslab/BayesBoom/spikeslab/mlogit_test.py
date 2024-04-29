import unittest
import os

from BayesBoom.spikeslab import mlogit_spike
from BayesBoom.test_utils import find_project_root

import numpy as np
import pandas as pd


class MlogitTest(unittest.TestCase):
    def setUp(self):
        np.random.seed(8675309)

    def test_mcmc(self):
        dirname = os.path.join(find_project_root(), "BayesBoom", "spikeslab")
        fname = os.path.join(dirname, "autopref.txt")
        data = pd.read_csv(fname)

        model = mlogit_spike(
            data["type"],
            "age + sex + married",
            {},
            niter=100,
            data=data)
        self.assertIsInstance(model, mlogit_spike)


_debug_mode = False

if _debug_mode:
    import pdb  # noqa

    # Turn warnings into errors.
    # warnings.simplefilter("error")

    # Run the test you are trying to debug here.  Instantiate the test class,
    # then call the problematic test.  Call pdb.pm() in the event of an
    # exception.
    print("Hello, world!")

    rig = MlogitTest()

    if hasattr(rig, "setUpClass"):
        rig.setUpClass()
    if hasattr(rig, "setUp"):
        rig.setUp()

    rig.test_mcmc()

    print("Goodbye, cruel world!")

if __name__ == "__main__":
    unittest.main()
