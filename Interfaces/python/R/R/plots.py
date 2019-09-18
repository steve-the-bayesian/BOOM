import matplotlib.pyplot as plt
import numpy as np


def plot_dynamic_distribution(
        curves,
        timestamps=None,
        quantile_step=.1,
        xlim=None,
        ylim=None,
        xlab="Time",
        ylab="distribution",
        col="black",
        ax=None,
        show=True,
        **kwargs):
    """Plot a dynamic distribution represented by a collection of curves simulated
    from that distribution.

    Args:
      curves:
        A numpy matrix of time series.  Rows correspond to different series,
        and columns correspond to time.

      timestamps:
        An array-like collection of increasing time stamps corresponding to the
        time points in 'curves.'

      quantile_step:
        The plotted distribution is formed by taking the quantiles of the
        curves at each time point.  The smaller the value of quantile_step the
        finer the approximation, but the larger and slower the plot.

      xlim:
        The limits on the horizontal axis.

      xlab:
        The label for the horizontal axis.

      ylim: the y limits (y1, y2) of the plot.
      ylab: Label for the vertical axis.
      **kwargs: Extra arguments passed to .... ???

    """

    if ax is None:
        fig, ax = plt.subplots(1, 1, **kwargs)

    quantile_points = np.arange(0, 1, quantile_step)
    curve_quantiles = np.quantile(curves, q=quantile_points, axis=0)
    if timestamps is None:
        timestamps = np.arange(curves.shape[1])
    assert(len(timestamps) == curves.shape[1])

    for i in range(int(np.floor(len(quantile_points) / 2))):
        lo = curve_quantiles[i, :]
        hi = curve_quantiles[-1-i, :]
        ax.fill_between(timestamps, lo, hi,
                        color=col,
                        facecolor="none",
                        edgecolor="none",
                        lw=.0000,
                        alpha=(i / len(quantile_points)))

    if xlim is not None:
        ax.set_xlim(xlim)

    if ylim is not None:
        ax.set_ylim(ylim)

    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)

    if show:
        fig.show()

    return ax
