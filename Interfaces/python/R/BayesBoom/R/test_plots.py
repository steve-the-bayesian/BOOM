import unittest
import matplotlib.pyplot as plt
<<<<<<< HEAD
import numpy as np
import pandas as pd
from BayesBoom.R.plots import (
    mosaic_plot,
    plot_dynamic_distribution
)


def char_range(c1, c2):
    """Generates the characters from `c1` to `c2`, inclusive."""
    for c in range(ord(c1), ord(c2) + 1):
        yield chr(c)


LETTERS = list(char_range("a", "z"))

us_states = ["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DC", "DE", "FL", "GA",
             "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA",
             "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY",
             "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX",
             "UT", "VT", "VA", "WA", "WV", "WI", "WY"]
=======
>>>>>>> ce1ee6781bdc0cd07a488a5f004e969bc849e1c4

from BayesBoom.R.plots import (
    plot_dynamic_distribution,
    #    compare_dynamic_distributions,
    time_series_boxplot,
)

import BayesBoom.R as R

import numpy as np
import pandas as pd


class TestPlots(unittest.TestCase):

    def setUp(self):
        pass

    def test_plot_points(self):
        x = np.sort(np.random.randn(100))
        y = 3 + .4 * x + np.random.randn(100) * .2
        fig, ax = plt.subplots()
        ax.scatter(x, y)
        if _show_figs:
            fig.show()

    def test_fill_between(self):
        x = np.sort(np.random.randn(100))
        y = np.abs(3 + .4 * x + np.random.randn(100) * .2)
        negy = -1 * y
        fig, ax = plt.subplots()
        ax.fill_between(x, y, negy, color="black", edgecolor="none")
        if _show_figs:
            fig.show()

    def test_plot_dynamic_distribution(self):
        time_dimension = 50
        niter = 500
        noise = np.random.randn(niter, time_dimension)
        random_walks = np.cumsum(noise, axis=1)

        timestamps = np.arange(0, time_dimension)
        fig, ax = plt.subplots()

        plot_dynamic_distribution(curves=random_walks, timestamps=timestamps,
                                  ax=ax, quantile_step=.01)
        if _show_figs:
            fig.show()

    def test_mosaic_plot(self):
        counts = np.random.randint(1, 10, (12, 4))
        counts = pd.DataFrame(counts, index=us_states[:counts.shape[0]],
                              columns=["red", "blue", "green", "yellow"])

        print(counts)
        fig, ax = plt.subplots()
        foo = mosaic_plot(counts, ax=ax)

        if _debug_mode:
            fig.show()

        self.assertIsInstance(foo, plt.Axes)

        if _show_figs:
            fig.show()

    def test_time_series_boxplot(self):
        x = np.random.randn(100, 8)
        time = pd.date_range(start="2020-02-1", periods=8, end="2020-02-08")
        fig, ax = plt.subplots()
        time_series_boxplot(x, time, ax=ax)
        if _show_figs:
            fig.show()

    def test_lty(self):
        x = np.linspace(0, 10)
        fig, ax = plt.subplots()
        for i in range(10):
            y = x + i
            ax.plot(x, y, ls=R.lty(i))
        if _show_figs:
            fig.show()


_debug_mode = False
_show_figs = _debug_mode


if _debug_mode:
    import pdb  # noqa

    # Turn warnings into errors.
    # warnings.simplefilter("error")

    # Run the test you are trying to debug here.  Instantiate the test class,
    # then call the problematic test.  Call pdb.pm() in the event of an
    # exception.
    print("Hello, world!")

    rig = TestPlots()
    if hasattr(rig, "setUpClass"):
        rig.setUpClass()
    if hasattr(rig, "setUp"):
        rig.setUp()

    # rig.test_plot_points()
<<<<<<< HEAD
    rig.test_mosaic_plot()
=======
    rig.test_lty()
>>>>>>> ce1ee6781bdc0cd07a488a5f004e969bc849e1c4
    # rig.test_fill_between()

    print("Goodbye, cruel world!")

else:
    if __name__ == "__main__":
        unittest.main(verbosity=2)
