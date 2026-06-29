import unittest
import BayesBoom.R as R
import BayesBoom.boom as boom
import numpy as np
import matplotlib.pyplot as plt

        
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
