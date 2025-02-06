import unittest
import numpy as np
import pandas as pd
import pickle

# pylint: disable=unused-import
import sys
import pdb

import matplotlib.pyplot as plt

import BayesBoom.R as R

import BayesBoom.mixtures as mix

class TestHmm(unittest.TestCase):
    def setUp(self):
        self._fetal_lamb_data = np.array(
            [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1,
             0, 1, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0,
             1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 1, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 7, 3, 2,
             3, 2, 4, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 2, 0, 0, 0, 0, 1, 0,
             0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
             2, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 2,
             0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 2, 0, 1, 2, 1, 1, 2, 1, 0, 1, 1, 0, 0,
             1, 1, 0, 0, 0, 1, 1, 1, 0, 4, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0], dtype='int')

    def test_fetal_lamb(self):
        hmm = mix.HiddenMarkovModel(2)
        hmm.add_data(self._fetal_lamb_data)


        lower_poisson_model = R.PoissonModel()
        lower_poisson_model.set_prior(R.GammaModel(1.0, 1.0))

        upper_poisson_model = R.PoissonModel()
        upper_poisson_model.set_prior(R.GammaModel(2.0, 1.0))
        
        hmm.add_state_model(lower_poisson_model)
        hmm.add_state_model(upper_poisson_model)

        niter = 100
        hmm.train(niter=niter, ping=None)

        self.assertEqual(hmm._log_likelihood_draws.shape[0], niter)
        self.assertEqual(hmm.markov_model._transition_probability_draws.shape[0],
                         niter)
        self.assertEqual(lower_poisson_model._lambda_draws.shape[0],
                         niter)
        self.assertEqual(upper_poisson_model._lambda_draws.shape[0],
                         niter)
        
    
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

    rig = TestHmm()

    if hasattr(rig, "setUpClass"):
        rig.setUpClass()
    if hasattr(rig, "setUp"):
        rig.setUp()

    rig.test_fetal_lamb()

    print("Goodbye, cruel world!")

else:
    if __name__ == "__main__":
        unittest.main(verbosity=2)
