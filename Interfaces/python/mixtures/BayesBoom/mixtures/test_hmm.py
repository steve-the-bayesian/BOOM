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
        print("testing fetail lamb data with Poisson models.")

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


    def test_categorical(self):

        # simulate some Categorical data.

        initial_distribution = np.array([.8, .2])
        transition_probabilities = np.array([[.9, .1],
                                             [.2, .8]])

        sample_size = 100
        markov = R.MarkovModel(transition_probabilities, initial_distribution)
        states = markov.sim(sample_size)

        colors = ["Red", "Blue", "Green"]
        first = R.MultinomialModel(np.array([.5, .3, .2]), categories=colors)
        second = R.MultinomialModel(np.array([.2, .3, .5]), categories=colors)

        values = ["blah"] * sample_size
        for i in range(sample_size):
            if states[i] == 0:
                values[i] = first.sim(1)[0]
            else:
                values[i] = second.sim(1)[0]

        learned1 = R.MultinomialModel(np.ones(3) / 3, categories=colors)
        learned2 = R.MultinomialModel(np.ones(3) / 3, categories=colors)

        hmm = mix.HiddenMarkovModel(2)
        hmm.add_data(values)

        hmm.add_state_model(learned1)
        hmm.add_state_model(learned2)

        niter = 100
        hmm.train(niter=niter, ping=None)

    def test_multilevel_categorical(self):
        taxonomy = [
            "red/crimson",
            "red/brick",
            "blue/sky",
            "blue/ocean",
        ]

        initial_distribution = np.array([.8, .2])
        transition_probabilities = np.array([[.9, .1],
                                             [.2, .8]])

        sample_size = 100
        markov = R.MarkovModel(transition_probabilities, initial_distribution)
        states = markov.sim(sample_size)

        p0 = np.array([.1, .2, .3, .4])
        p1 = np.array([.4, .3, .2, .1])
        data = np.array([""] * sample_size, dtype="object")
        n0 = np.sum(states == 0)
        n1 = sample_size - n0
        data[states == 0] = np.random.choice(taxonomy, size=n0, replace=True, p=p0)
        data[states == 1] = np.random.choice(taxonomy, size=n1, replace=True, p=p1)

        boom_taxonomy = boom.Taxonomy(taxonomy, "/")
        state1 = R.MultilevelMultinomialModel(boom_taxonomy)
        state2 = R.MultilevelMultinomialModel(boom_taxonomy)

        hmm = mix.HiddenMarkovModel(2)
        hmm.add_data(data)
        hmm.add_state_model(state1)
        hmm.add_state_model(state2)

        hmm.train(100)

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
