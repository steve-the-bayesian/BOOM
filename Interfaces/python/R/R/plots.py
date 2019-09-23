import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import numbers

from R import data_range

_last_figure_ = None
_last_axis_ = None


def plot_dynamic_distribution(
        curves,
        timestamps=None,
        quantile_step=.1,
        xlab="Time",
        ylab="distribution",
        col="black",
        ax=None,
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

      **kwargs: Extra arguments passed to .... ???

    """
    fig = None
    if ax is None:
        fig, ax = plt.subplots(1, 1)

    plot_options, kwargs = _skim_plot_options(**kwargs)

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

    _set_plot_options(**plot_options)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    global _last_axis_
    _last_axis_ = ax
    return ax


def hosmer_lemeshow_plot(actual, predicted, ax=None, **kwargs):
    """Construct a Hosmer Lemeshow plot on the supplied axes.  A Hosmer Lemeshow
    plot partitions the predicted values into groups, and compares the
    midpoint of each group with the observed proportion in the group.

    Args:
      actual: The actual set of 0's and 1's being modeled.
      predicted: The predicted probabilities coresponding to 'actual'.
      ax: The pyplot axes on which to draw the plot, or None.  If None the
        figure will be shown on function exit.
      **kwargs: Passed to plt.subplots in the event that ax is None.

    Return:
      fig: If a new figure was created then this is the containing object.
        Otherwise None.
      ax: The axes object containing the plot.

    """
    fig = None
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    group_means = pd.DataFrame({"pred": predicted, "actual": actual}).groupby(
        pd.qcut(predicted, 10))["actual"].mean()
    bar_locations = group_means.index.categories.mid.values
    lower = np.array([x.left for x in group_means.index.values])
    upper = np.array([x.right for x in group_means.index.values])
    bar_widths = .8 * (upper - lower)

    ax.barh(bar_locations, group_means, height=bar_widths)
    ax.set_xticks(bar_locations)
    labels = [str(lab) for lab in group_means.index.values]
    ax.set_yticklabels(labels)
    ax.set_ylabel("Predicted Proportions")

    xticks = pretty_plot_ticks(np.min(predicted), np.max(predicted), 5)
    ax.set_xticks(xticks)
    ax.set_xlabel("Observed Proportions")

    global _last_axis_
    _last_axis_ = ax
    return fig, ax


def plot_many_ts(series, same_scale=True, ylim=None, gap=0, truth=None,
                 **kwargs):
    nseries = series.shape[1]
    nr = int(max(1, np.sqrt(nseries)))
    nc = int(np.ceil(nseries / nr))
    gap *= .2
    fig, ax = plt.subplots(nr, nc)
    fig.subplots_adjust(hspace=gap, wspace=gap)

    series_number = 0
    if same_scale:
        ylim = data_range(series)

    if truth is not None:
        if isinstance(truth, numbers.Number):
            truth = np.full(nseries, truth)

    for i in range(nr):
        for j in range(nc):
            ax[i, j].set_frame_on(True)
            ax[i, j].xaxis.set_tick_params(bottom=False, top=False,
                                           labelbottom=False, labeltop=False)
            ax[i, j].yaxis.set_tick_params(left=False, right=False,
                                           labelleft=False, labelright=False)

            # In the top and bottom row, set x axis to be either the bottom or
            # the top.
            if i == 0:
                if (j % 2) == 1:
                    ax[i, j].xaxis.set_tick_params(
                        top=True, labeltop=True)

            if i + 1 == nr:
                if (j % 2) == 0:
                    ax[i, j].xaxis.set_tick_params(
                        bottom=True, labelbottom=True)

            if same_scale:
                # In the left or right column, set the y axis to be either the
                # left or the right.
                ax[i, j].set_ylim(ylim)
                if j == 0 and (i % 2) == 0:
                    ax[i, j].yaxis.set_tick_params(left=True, labelleft=True)
                if j + 1 == nc and (i % 2) == 1:
                    ax[i, j].yaxis.set_tick_params(right=True, labelright=True)
                    ax[i, j].yaxis.tick_right()

            else:
                if ylim is not None:
                    print("setting ylim")
                    ax[i, j].set_ylim(ylim)

            if series_number < nseries:
                ax[i, j].plot(series[:, series_number])
                if truth is not None:
                    abline(ax=ax[i, j], h=truth[series_number])
                series_number += 1

    return fig, ax


def barplot(x, labels=None, zero=True, ax=None, **kwargs):
    """Make a horizonal bar plot.
    Args:
      x:  Array-like collection of numbers to plot.
      labels: Labels for the bars.  If x is a pd.Series then labels==None means
        to take the labels from the series index.  Otherwise None means don't
        plot labels.
      zero:  Bool.  Should zero be forcibly included in the numeric axis.
      xlab: Label for the horizonal "numeric" axis.
      ylab: Label for the vertical "categorial" axis.
      kwargs: extra arguments passed to plt.subplots or plt.barh.

    """
    fig = None
    if ax is None:
        fig, ax = plt.subplots(1, 1)

    x = x[::-1]
    if labels is not None:
        labels = labels[::-1]

    if labels is None and isinstance(x, pd.Series):
        labels = [str(lab) for lab in x.index]
    bar_locations = np.arange(len(x))
    plot_options, kwargs = _skim_plot_options(**kwargs)

    ax.barh(bar_locations, x, height=.8, **kwargs)
    ax.set_yticks(bar_locations)
    ax.set_yticklabels(labels)
    lo = np.nanmin(x)
    hi = np.nanmax(x)
    if zero and lo > 0:
        lo = 0.0
    if zero and hi < 0:
        hi = 0.0

    ax.set_xticks(pretty_plot_ticks(lo, hi, 5))
    _set_plot_options(ax, **plot_options)
    global _last_axis_
    _last_axis_ = ax
    return fig, ax


def plot(x, y, ax=None, **kwargs):
    # TODO: make this generic
    fig = None
    if ax is None:
        fig, ax = plt.subplots(1, 1)

    plot_options, kwargs = _skim_plot_options(**kwargs)

    ax.scatter(x, y, **kwargs)
    _set_plot_options(ax, **plot_options)

    global _last_axis_
    _last_axis_ = ax
    return fig, ax


def points(x, y, ax=None, **kwargs):
    if ax is None:
        global _last_axis_
        ax = _last_axis_
    ax.scatter(x, y, **kwargs)


def lines(x, y, ax=None, **kwargs):
    if ax is None:
        global _last_axis_
        ax = _last_axis_
    ax.plot(x, y, **kwargs)


def hist(x, ax=None, **kwargs):
    fig = None
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    plot_options, kwargs = _skim_plot_options(**kwargs)
    ax.hist(x, **kwargs)
    _set_plot_options(ax, **plot_options)
    global _last_axis_
    _last_axis_ = ax
    return fig, ax


def _skim_plot_options(xlab="", ylab="", xlim=None, ylim=None, title="",
                       **kwargs):
    plot_options = {'xlab': xlab,
                    'ylab': ylab,
                    'xlim': xlim,
                    'ylim': ylim,
                    'title': title}
    return plot_options, kwargs


def _set_plot_options(ax, xlab="", ylab="", xlim=None, ylim=None, title="",
                      **kwargs):
    if xlab != "":
        ax.set_xlabel(xlab)
    if ylab != "":
        ax.set_ylabel(ylab)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    if title != "":
        ax.set_title(title)


def plot_ts(x, timestamps=None, ax=None, **kwargs):
    """ Plot a time series."""
    # if timestamps is None:
    #     if isinstance(x, pd.Series):
    fig = None
    if ax is None:
        fig, ax = plt.subplots(1, 1)

    if timestamps is None:
        if isinstance(x, pd.Series):
            timestamps = x.index

    ax.plot(x)
    _set_plot_options(ax, **kwargs)
    global _last_axis_
    _last_axis_ = ax
    return fig, ax


def boxplot(x, labels=None, ax=None, **kwargs):
    fig = None
    if ax is None:
        fig, ax = plt.subplots(1, 1)

    if labels is None and isinstance(x, pd.DataFrame):
        labels = x.columns

    ax.boxplot(x, vert=False)
    _set_plot_options(ax, **kwargs)
    global _last_axis_
    _last_axis_ = ax
    return fig, ax


def abline(a=0, b=1, h=None, v=None, ax=None, **kwargs):
    # TODO: documentation.
    # TODO: This does not work after calling show().
    global _last_axis_
    if ax is None:
        ax = _last_axis_
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    if h is not None:
        p0 = xlim[0], h
        p1 = xlim[1], h
    elif v is not None:
        p0 = v, ylim[0]
        p1 = v, ylim[1]
    else:
        p0 = xlim[0], a + b * xlim[0]
        p1 = xlim[1], a + b * xlim[1]

    x = [p0[0], p1[0]]
    y = [p0[1], p1[1]]
    ax.plot(x, y)


def pretty_plot_ticks(low, high, n):
    """Return a set of 'pretty' tick labels.

    Args:
      low: The lower end of the plotting range.
      high: The upper end of the plotting range.
      n: The desired number of ticks.

    Returns:
      A np.array with 'pretty' values suitable for tick labels.  The length of
      the array will typically not be exactly 'n'.  Its lower endpoint will be
      <= low and its upper endpoint will be >= high.

    Taken from StackOverflow:
    https://stackoverflow.com/questions/43075617/python-function-equivalent-to-rs-pretty

    """
    def nicenumber(x, round):
        exp = np.floor(np.log10(x))
        f = x / 10**exp

        if round:
            if f < 1.5:
                nf = 1.
            elif f < 3.:
                nf = 2.
            elif f < 7.:
                nf = 5.
            else:
                nf = 10.
        else:
            if f <= 1.:
                nf = 1.
            elif f <= 2.:
                nf = 2.
            elif f <= 5.:
                nf = 5.
            else:
                nf = 10.

        return nf * 10.**exp

    range = nicenumber(high - low, False)
    d = nicenumber(range / (n - 1), True)
    miny = np.floor(low / d) * d
    maxy = np.ceil(high / d) * d
    return np.arange(miny, maxy+0.5*d, d)
