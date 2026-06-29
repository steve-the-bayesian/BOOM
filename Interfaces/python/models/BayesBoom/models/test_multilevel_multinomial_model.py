import unittest
import BayesBoom.R as R
import BayesBoom.boom as boom
import numpy as np
import matplotlib.pyplot as plt

class TestMultilevelMultinomialModel(unittest.TestCase):
    def setUp(self):
        np.random.seed(8675309)
        boom.GlobalRng.rng.seed(8675309)
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
        
