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
        component0 = R.NormalPrior(0, 1)
        component0.set_prior(R.NormalInverseGammaModel(
            0.0, 1.0, 1.0, 1.0))
        model.add_component(component0)

        component1 = R.NormalPrior(0, 1)
        component1.set_prior(R.NormalInverseGammaModel(
            0.0, 1.0, 1.0, 1.0))
        model.add_component(component1)

        model.add_data(y)

        niter = 1000
        model.train(niter=niter)
        self.assertEqual(
            len(model._mixture_components[0]._mu_draws),
            niter)
        self.assertEqual(
            len(model._mixture_components[0]._sigma_draws),
            niter)

        self.assertEqual(model._mixing_distribution._prob_draws.shape[0],
                         niter)

        fig, ax = model.plot_components()
        self.assertIsInstance(fig, plt.Figure)


    def test_markov(self):
        P0 = np.array([[.8, .2],
                       [.1, .9]])
        P1 = np.array([[.3333, .6667],
                       [.3333, .6667]])
        pi0 = np.array([.5, .5])
        mixing_weights = np.array([.65, .35])
        num_users = 50
        sample_size_per_user = 20
        
        suf_list = []
        for user in range(num_users):
            u = np.random.rand()
            if u < mixing_weights[0]:
                sim_model = R.MarkovModel(P0, pi0)
            else:
                sim_model = R.MarkovModel(P1, pi0)
            data = sim_model.sim(sample_size_per_user)
            suf_list.append(R.MarkovSuf(data))


        m0 = R.MarkovModel(state_size=2) 
        m1 = R.MarkovModel(state_size=2)
        model = mix.FiniteMixtureModel()
        model.add_component(m0)
        model.add_component(m1)
        model.add_data(suf_list)

        niter = 1000
        import pdb
        pdb.set_trace()
        model.train(niter)
        
        

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
