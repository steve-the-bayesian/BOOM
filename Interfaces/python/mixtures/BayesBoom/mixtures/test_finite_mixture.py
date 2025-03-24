import unittest
import numpy as np
import pandas as pd
import pickle

# pylint: disable=unused-import
import sys
import pdb

import matplotlib.pyplot as plt

import BayesBoom.R as R
import BayesBoom.boom as boom
import BayesBoom.mixtures as mix

class TestFiniteMixture(unittest.TestCase):
    def setUp(self):
        np.random.seed(8675309)

    def test_gaussians(self):
        y0 = np.random.randn(50) + 3 + 7
        y1 = np.random.randn(100) * 7 + 3
        y = np.concat((y0, y1))

        model = mix.FiniteMixtureModel()
        model.add_component(R.NormalPrior(0, 1))
        model.add_component(R.NormalPrior(0, 1))

        model.add_data(y)

        model.train(niter=1000)
        

_debug_mode = False

if _debug_mode:
    import pdb  # noqa

# Turn warnings into errors.
#    import warnings
#    warnings.simplefilter("error")

    # Run the test you are trying to debug here.  Instantiate the test class,
    # then call the problematic test.  Call pdb.pm() in the event of an
    # exception.
    print("Hello, world!")

    rig = TestFiniteMixture()

    if hasattr(rig, "setUpClass"):
        rig.setUpClass()
    if hasattr(rig, "setUp"):
        rig.setUp()

    rig.test_fetal_lamb()

    print("Goodbye, cruel world!")

else:
    if __name__ == "__main__":
        unittest.main(verbosity=2)
