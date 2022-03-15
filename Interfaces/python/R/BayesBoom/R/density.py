import numpy as np
from .plots import _skim_plot_options, _set_plot_options


class Density:
    """
    A Gaussian kernel density estimate of a scalar valued variable.
    """

    def __init__(self, x, bw: float = None):
        """
        Args:
          x: a 1-d numpy array or object convertible to such.
          bw:  None means to use R's default bandwidth.
        """
        from scipy.stats import gaussian_kde

        x = np.array(x).ravel()
        finite = np.isfinite(x)
        x = x[finite]
        x.sort()
        if bw is None:
            bw = self._default_bandwidth(x)

        self._kde = gaussian_kde(x)
        self._grid = np.linspace(x[0], x[-1], 100)
        self._density_values = self(self._grid)

    def plot(self, ax=None, **kwargs):
        """
        Plot the density function on the supplied set of Axes.
        """
        import matplotlib.pyplot as plt

        fig = None
        if ax is None:
            fig, ax = plt.subplots(1, 1)

        plot_options, kwargs = _skim_plot_options(**kwargs)
        ax.plot(self._grid, self._density_values, **kwargs)
        _set_plot_options(ax, **plot_options)
        if fig is not None:
            fig.show()
        return ax

    def __call__(self, x):
        """
        Treat a density object as function to be evaluated at x.
        """
        return self._kde(x)

    @staticmethod
    def _default_bandwidth(x):
        """
        R's default logic for selecting the bandwidth of a kernel density
        estimator for x.  In R this function is called 'bw.nrd0'.
        """
        sdx = np.std(x, ddof=1)
        quantiles = np.quantile(x, (.25, .75))
        iqr = np.max(quantiles) - np.min(quantiles)

        hi = sdx
        lo = min(hi, iqr / 1.34)
        if lo <= 0.0:
            lo = hi
            if lo <= 0.0:
                lo = abs(x[0])
                if lo <= 0.0:
                    lo = 1
        return .9 * lo * len(x) ** -.2
