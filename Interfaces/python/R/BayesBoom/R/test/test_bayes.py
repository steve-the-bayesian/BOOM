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
        self.taxonomy = [
            "red/crimson",
            "red/brick",
            "blue/sky",
            "blue/ocean",
        ]


    def test_model(self):
        model = R.MultilevelMultinomialModel(self.taxonomy)
        top_probs = model.probs()
        self.assertEqual(2, len(top_probs))
        red_probs = model.probs("red")
        self.assertEqual(2, len(red_probs))
        blue_probs = model.probs("blue")
        self.assertEqual(2, len(blue_probs))

        niter = 1000
        model.allocate_space(niter)
        self.assertIsInstance(model._model_levels, list)
        self.assertEqual(len(model._model_levels), 3)
        self.assertEqual(model._model_levels[0], "")
        self.assertTrue("red" in model._model_levels)
        self.assertTrue("blue" in model._model_levels)

        self.assertIsInstance(model._draws, dict)
        for x in model._model_levels:
            self.assertTrue(x in model._draws.keys())
        
        data = np.random.choice(self.taxonomy, size=100, replace=True)
        data_builder = model.create_boom_data_builder()
        boom_data = data_builder.build_boom_data(data)
        self.assertIsInstance(boom_data, list)
        self.assertEqual(len(boom_data), len(data))
        self.assertIsInstance(boom_data[0], boom.MultilevelCategoricalData)

        for data_point in boom_data:
            model._boom_model.add_data(data_point)
        
        for i in range(niter):
            model._boom_model.sample_posterior()
            model.record_draw(i)

        self.assertEqual(model._boom_taxonomy.child_levels(""),
                         ["blue", "red"])
        self.assertEqual(model._boom_taxonomy.child_levels("red"),
                         ["brick", "crimson"])
        self.assertEqual(model._boom_taxonomy.child_levels("blue"),
                         ["ocean", "sky"])
        
        self.assertEqual(model._boom_taxonomy.pop_level("red"),
                         ("", "red"))
        
        top_prob_draws = model.prob_draws("")
        red_prob_draws = model.prob_draws("red")
        blue_prob_draws = model.prob_draws("blue")
        red_conditional_prob_draws = model.prob_draws("red", conditional=True)
        blue_conditional_prob_draws = model.prob_draws("blue", conditional=True)
        R.compare_den(red_prob_draws)

    def test_mixture_plot(self):
        probs1 = np.array([.4, .3, .2, .1])
        probs2 = np.array([.1, .2, .3, .4])

        model1 = R.MultilevelMultinomialModel(self.taxonomy)
        model1.boom()
        data_builder = model1.create_boom_data_builder()
        data1 = np.random.choice(self.taxonomy, size=100, p=probs1, replace=True)
        model1_data = data_builder.build_boom_data(data1)
        for data_point in model1_data:
            model1._boom_model.add_data(data_point)
            
        model2 = R.MultilevelMultinomialModel(self.taxonomy)
        model2.boom()
        data2 = np.random.choice(self.taxonomy, size=50, p=probs2, replace=True)
        model2_data = data_builder.build_boom_data(data2)
        for data_point in model2_data:
            model2._boom_model.add_data(data_point)
        
        niter = 1000
        model1.allocate_space(niter)
        model2.allocate_space(niter)
        for i in range(niter):
            model1._boom_model.sample_posterior()
            model1.record_draw(i)
            model2._boom_model.sample_posterior()
            model2.record_draw(i)
            
        fig_ts, ax_ts = model1.plot_components([model1, model2], style="ts")
        fig_ts.suptitle("MultilevelCategoricalData Top Level Plot")
        fig_box, ax_box = model1.plot_components([model1, model2], style="box")
        fig_box.suptitle("MultilevelCategoricalData Top Level Plot")
        fig_den, ax_den = model1.plot_components([model1, model2], style="den")
        fig_den.suptitle("MultilevelCategoricalData Top Level Plot")
        fig_bar, ax_bar = model1.plot_components([model1, model2], style="bar")
        fig_bar.suptitle("MultilevelCategoricalData Top Level Plot")

        
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
