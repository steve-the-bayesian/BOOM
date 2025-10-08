import unittest
import BayesBoom.R as R
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class TestPlotMultilevelProbabilities(unittest.TestCase):

    def setUp(self):
        np.random.seed(8675309)
        self.taxonomy = [
            "red/crimson",
            "red/brick",
            "blue/sky",
            "blue/ocean",
        ]

    def test_plot(self):
        probs = []
        probs.append(pd.Series(np.array([.6, .3]), index=["red", "brick"]))
        probs.append(pd.Series(np.array([.6, .7]), index=["red", "crimson"]))
        probs.append(pd.Series(np.array([.4, .8]), index=["blue", "sky"]))
        probs.append(pd.Series(np.array([.4, .2]), index=["blue", "ocean"]))

        bar_lengths = np.array([
            .6 * .3,
            .6 * .7,
            .4 * .8,
            .4 * .2,
        ])
        self.assertAlmostEqual(np.sum(bar_lengths), 1.0)
        
        colors = ["blue", "red"]
#         figp, axp = R.plot_multilevel_probabilities(probs, colors=colors, logscale=False)
#         fig, ax = R.plot_multilevel_probabilities(probs, colors=colors, logscale=True)
        # plt.show()

    def test_bar_info(self):
        probs = []
        probs.append(pd.Series(np.array([.6, .3]), index=["red", "brick"]))
        probs.append(pd.Series(np.array([.6, .7]), index=["red", "crimson"]))
        probs.append(pd.Series(np.array([.4, .8]), index=["blue", "sky"]))
        probs.append(pd.Series(np.array([.4, .2]), index=["blue", "ocean"]))

        info = R.BarInfo()
        for prob in probs:
            info.add_bar(prob)

        fig, ax = info.plot(None, None)
        import pdb
        pdb.set_trace()
        print("blah")
        

        
_debug_mode = False

if _debug_mode:
    import pdb  # noqa

    # Turn warnings into errors.
    # warnings.simplefilter("error")

    # Run the test you are trying to debug here.  Instantiate the test class,
    # then call the problematic test.  Call pdb.pm() in the event of an
    # exception.
    print("Hello, world!")

    rig = TestPlotMultilevelProbabilities()
    if hasattr(rig, "setUpClass"):
        rig.setUpClass()
    if hasattr(rig, "setUp"):
        rig.setUp()

    rig.test_plot()

    if hasattr(rig, "tearDown"):
        rig.tearDown()

    print("Goodbye, cruel world!")

else:
    if __name__ == "__main__":
        unittest.main(verbosity=2)
