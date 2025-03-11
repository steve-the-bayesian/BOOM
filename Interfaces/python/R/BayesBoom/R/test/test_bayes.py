import unittest
import BayesBoom.R as R
import BayesBoom.boom as boom
import numpy as np
import matplotlib.pyplot as plt


class TestRegSuf(unittest.TestCase):

    def setUp(self):
        np.random.seed(8675309)
        sample_size = 1000
        xdim = 4
        self._x = np.random.randn(sample_size, xdim)
        self._x[:, 0] = 1.0
        self._y = np.random.randn(sample_size)

    def test_eval(self):
        xtx = self._x.T @ self._x
        xty = self._x.T @ self._y
        sdy = np.std(self._y, ddof=1)
        sample_size = self._y.size
        ybar = self._y.mean()
        xbar = self._x.mean(axis=0)
        suf = R.RegSuf(xtx, xty, sdy, sample_size, ybar, xbar)
        yty = np.sum(self._y ** 2)

        bs = suf.boom()
        self.assertTrue(np.allclose(bs.xtx.to_numpy(), xtx))
        self.assertTrue(np.allclose(bs.xty.to_numpy(), xty))
        self.assertEqual(bs.sample_size, sample_size)
        self.assertAlmostEqual(bs.yty, yty, 4)


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

        
class TestPoissonModel(unittest.TestCase):
    def setUp(self):
        np.random.seed(8675309)

    def test_poisson(self):
        model = R.PoissonModel(2.3)
        boom_model = model.boom()
        self.assertIsInstance(boom_model, boom.PoissonModel)

    def test_plots(self):
        S = 3
        niter = 1000
        models = []
        for s in range(S):
            model = R.PoissonModel(1.0)
            model._lambda_draws = np.random.randn(niter)**2
            models.append(model)
        fig_ts, ax_ts = models[0].plot_components(models)
        fig_den, ax_den = models[0].plot_components(models, style="den")
        fig_box, ax_box = models[0].plot_components(models, style="box")


class TestMultinomialModel(unittest.TestCase):

    def setUp(self):
        np.random.seed(8675309)

    def test_plots(self):
        models = []
        models.append(R.MultinomialModel(np.array([.1, .3, .6])))
        models.append(R.MultinomialModel(np.array([.3, .4, .3])))
        models.append(R.MultinomialModel(np.array([.6, .2, .2])))

        models[0]._prob_draws = np.random.dirichlet(np.array([100, 300, 600]), size=1000)
        models[1]._prob_draws = np.random.dirichlet(np.array([300, 400, 300]), size=1000)
        models[2]._prob_draws = np.random.dirichlet(np.array([600, 200, 200]), size=1000)

        models[0].plot_components(models)
        models[0].plot_components(models, style="box")
        models[0].plot_components(models, style="bar")
    

class TestMultilevelMultinomialModel(unittest.TestCase):
    def setUp(self):
        np.random.seed(8675309)

    def test_model(self):
        taxonomy = [
            "red/crimson",
            "red/brick",
            "blue/sky",
            "blue/ocean",
        ]

        model = R.MultilevelMultinomialModel(taxonomy)
        top_probs = model.probs()
        self.assertEqual(2, len(top_probs))
        red_probs = model.probs("red")
        self.assertEqual(2, len(red_probs))
        blue_probs = model.probs("blue")
        self.assertEqual(2, len(blue_probs))

        model.allocate_space(5)
        self.assertIsInstance(model._model_levels, list)
        self.assertEqual(len(model._model_levels), 3)
        self.assertEqual(model._model_levels[0], "top")
        self.assertTrue("red" in model._model_levels)
        self.assertTrue("blue" in model._model_levels)

        self.assertIsInstance(model._draws, dict)
        for x in model._model_levels:
            self.assertTrue(x in model._draws.keys())
        
        model.record_draw(3)

        data = np.random.choice(taxonomy, size=100, replace=True)
        data_builder = model.create_boom_data_builder()
        boom_data = data_builder.build_boom_data(data)
        self.assertIsInstance(boom_data, list)
        self.assertEqual(len(boom_data), len(data))
        self.assertIsInstance(boom_data[0], boom.MultilevelCategoricalData)

        
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
