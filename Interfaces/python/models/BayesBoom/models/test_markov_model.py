import unittest
import BayesBoom.R as R
import BayesBoom.boom as boom
import numpy as np
import matplotlib.pyplot as plt


class TestMarkovModel(unittest.TestCase):

    def setUp(self):
        np.random.seed(8675309)

    def test_model(self):
        probs = np.ones((3, 3)) / 3.0
        init = np.ones(3) / 3.0

        model = R.MarkovModel(probs, init)
        boom_model = model.boom()
        self.assertIsInstance(boom_model, boom.MarkovModel)

        probs = model.transition_probabilities
        self.assertEqual(probs.shape, (3, 3))
        self.assertTrue(np.allclose(
            probs.sum(axis=1),
            np.ones(3)))

    def test_sim(self):
        trans_probs = np.array([[.9, .1],
                                [.15, .85]])
        init = np.array([.7, .3])
        model = R.MarkovModel(trans_probs, init)
        chain = model.sim(10)
        self.assertEqual(len(chain), 10)

    def test_plots(self):
        models = []
        num_mix = 2
        state_size = 3
        for i in range(num_mix):
            probs = np.random.randn(100, state_size, state_size)
            probs = np.abs(probs)
            totals = probs.sum(axis=2)
            for s in range(state_size):
                probs[:, :, s] = probs[:, :, s] / totals
            # probs = probs / totals
            model = R.MarkovModel(probs[0, :, :])
            model._transition_probability_draws = probs
            models.append(model)

        fig_ts, ax_ts = models[0].plot_components(models)
        fig_box, ax_box = models[0].plot_components(models, style="box")
        fig_den, ax_den = models[0].plot_components(models, style="den")

        
_debug_mode = False

if _debug_mode:
    import pdb  # noqa

    # Turn warnings into errors.
    # warnings.simplefilter("error")

    # Run the test you are trying to debug here.  Instantiate the test class,
    # then call the problematic test.  Call pdb.pm() in the event of an
    # exception.
    print("Hello, world!")

    rig = TestRegSuf()
    if hasattr(rig, "setUpClass"):
        rig.setUpClass()
    if hasattr(rig, "setUp"):
        rig.setUp()

    rig.test_eval()

    print("Goodbye, cruel world!")

else:
    if __name__ == "__main__":
        unittest.main(verbosity=2)
