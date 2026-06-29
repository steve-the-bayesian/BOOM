import unittest
import BayesBoom.R as R
import BayesBoom.boom as boom
import numpy as np
import matplotlib.pyplot as plt


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
        
