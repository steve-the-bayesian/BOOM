import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from numbers import Number
from abc import ABC, abstractmethod

from .R import (
    data_range,
    remove_common_suffix,
    remove_common_prefix,
    unique_match
)

_active_graphics_devices = {}
_current_graphics_device = None
_largest_graphics_device_number = 0


class GraphicsDevice(ABC):
    """
    Manages a plt.figure and a set of axes.
    """

    def __init__(self):
        """
        Creating a new graphics device updates the global set of devices
        """
        global _largest_graphics_device_number
        global _current_graphics_device
        global _active_graphics_devices
        self._number = _largest_graphics_device_number + 1
        _largest_graphics_device_number = self._number
        _active_graphics_devices[self._number] = self
        _current_graphics_device = self

        self._figure, self._axes = plt.subplots(1, 1)
        self._axes_cursor = None
        self._nrow = 1
        self._ncol = 1

    def __del__(self):
        plt.close(self._figure)
        global _active_graphics_devices
        del _active_graphics_devices[self._number]

    @property
    def next_axes(self):
        """
        The next set of axes on which a plot is to be drawn.  If the last set of
        available axes has been exhausted, the stored Figure is refreshed with
        new axes of the same dimension.
        """
        if hasattr(self._axes, "shape"):
            shape = self._axes.shape
            if min(shape) == 1:
                self._increment_1d_cursor()
            else:
                self._increment_2d_cursor()
        else:
            self._axes = self._figure.subplots(1, 1)
        return self.current_axes

    @property
    def current_axes(self):
        """
        The current set of axes.
        """
        if self._nrow > 1 or self._ncol > 1:
            return self._axes[self._axes_cursor]
        else:
            return self._axes

    @abstractmethod
    def draw_current_axes(self):
        """
        A hook to be called after the graphics device is updated.
        """

    def _create_subplots(self, nrow, ncol):
        """
        Set the graphics device to use multiple rows and/or columns of plots.
        """
        self._axes = self._figure.subplots(nrow, ncol)
        self._nrow = nrow
        self._ncol = ncol
        if min(nrow, ncol) > 1:
            self._axes_cursor = (0, 0)
        elif max(nrow, ncol) > 1:
            self._axes_cursor = 0
        else:
            self._axes_cursor = None

    def _increment_1d_cursor(self):
        """
        Increment the self._axes cursor when self._axes is a 1-d array.
        """
        dim = max(self._nrow, self._ncol)
        self._axes_cursor += 1
        if self._axes_cursor >= dim:
            self._axes_cursor = 0
            # We need nrow and ncol here because a 1-d axes might be a row, or
            # a column.
            self._axes = self._figure.subplots(self._nrow, self._ncol)

    def _increment_2d_cursor(self):
        """
        Increment self._axes_cursor when self._axes is two-dimensional.
        """
        i, j = self._axes_cursor
        j += 1
        if j >= self._axes.shape[1]:
            j = 0
            i += 1
            if i >= self._axes.shape[0]:
                self._axes = self._figure.subplots(
                    self._axes.shape[0], self._axes.shape[1])
                i, j = 0, 0
        self._axes_cursor = (i, j)


def dev_new():
    return InteractiveGraphicsDevice()


def dev_set(device_number: int):
    """
    Set the current graphics device to the device with the given number.

    """
    global _active_graphics_devices
    global _current_graphics_device
    device = _active_graphics_devices.get(device_number, None)
    if device is None:
        raise Exception(f"Graphics device {device_number} does note exist.")
    else:
        _current_graphics_device = device


class InteractiveGraphicsDevice(GraphicsDevice):
    """
    A private stack manages the set of active graphics devices.
    """

    def __init__(self):
        super().__init__()
        plt.show(block=False)
        plt.pause(.001)

    def draw_current_axes(self):
        """
        """
        plt.pause(.001)


class PdfGraphicsDevice(GraphicsDevice):
    """
    Destructor generates a PDF file that gets generated when the graphics
    device is deleted.
    """

    def __init__(self, filename, width=5, height=5):
        self._filename = filename

    def __del__(self):
        """
        Create the pdf file upon deletion.
        """
        if self._figure is not None:
            self._figure.save(self._filename)

    def draw_current_axes(self):
        """
        """
        pass


def get_current_graphics_device():
    """
    Returns the current graphics device, if one exists.  Otherwise create and
    return an interactive graphics device.
    """
    global _current_graphics_device
    if _current_graphics_device is not None:
        return _current_graphics_device
    else:
        return InteractiveGraphicsDevice()


# ===========================================================================
# Graphics related utilities.  These do not interact with a figure, axes, or
# graphics device.
# ===========================================================================
def _skim_plot_options(xlab="", ylab="", xlim=None, ylim=None, title="",
                       main="", **kwargs):
    """
    Remove plotting options used by the R package from other options supplied by
    kwargs.  This allows the remaining options to be later passed to pyplot.

    Returns:
      plot_options:  The options expected by R plotting functions.a
      kwargs:  Everything not pulled into plot_options.
    """
    plot_options = {'xlab': xlab,
                    'ylab': ylab,
                    'xlim': xlim,
                    'ylim': ylim,
                    'main': main,
                    'title': title}
    return plot_options, kwargs


def _set_plot_options(ax, xlab="", ylab="", xlim=None, ylim=None, title="",
                      main="", **kwargs):
    if xlab != "":
        ax.set_xlabel(xlab)
    if ylab != "":
        ax.set_ylabel(ylab)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    if main != "":
        title = main
    if title != "":
        ax.set_title(title)


def pretty_plot_ticks(low, high, n):
    """
    Return a set of 'pretty' tick locations.

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
    def nicenumber(x, round_result: bool):
        exp = np.floor(np.log10(x))
        f = x / 10**exp

        if round_result:
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

    num_range = nicenumber(high - low, False)
    d = nicenumber(num_range / (n - 1), True)
    miny = np.floor(low / d) * d
    maxy = np.ceil(high / d) * d
    return np.arange(miny, maxy+0.5*d, d)


def plot_grid_shape(nplots: int):
    """
    Compute the number of rows and columns needed to plot 'nplots'.

    :param nplots:
        The desired number of plots.

    :return tuple:
        The number of rows, and columns, needed to plot that many plots.
    """
    nr = int(max(1, np.sqrt(nplots)))
    nc = int(np.ceil(nplots / nr))
    return nr, nc


# ===========================================================================
# Low level plot functions.  These interact with the current axes object in the
# current graphics device.  They do not advance to the next axes.
# ===========================================================================
def points(x, y, s=None, **kwargs):
    """
    Add points to the plot showing on the current graphics device.
    """
    device = get_current_graphics_device()
    ax = device.current_axes()
    if s is None:
        s = 20 / np.sqrt(len(y))
    ax.scatter(x, y, **kwargs)


def lines(x, y, **kwargs):
    """
    Add lines to the most recent plot.
    """
    device = get_current_graphics_device()
    ax = device.current_axes()
    ax.plot(x, y, **kwargs)


def abline(a=0, b=1, h=None, v=None, ax=None, **kwargs):
    """
    Add a line with specified slope and intercept to a plot.

    Args:
      a: The intercept of the line.
      b: The slope of the line.
      h: Draw a horizontal line at this y value.
      v: Draw a vertical line at this x value.
      **kwargs:  Additional arguments passed to 'plt.plot'.

    Returns:
      None

    Effect:
      The requested line is plotted on 'ax'.
    """
    if ax is None:
        device = get_current_graphics_device()
        ax = device.current_axes

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
    ax.plot(x, y, **kwargs)


# ===========================================================================
# High level plot functions.  These advance to the next active axes object in
# the graphics device.  If the graphics device is already at the last set of
# axes, the plot resets.
# ===========================================================================

def plot(x, y=None, s=None, hexbin_threshold=1e+5, **kwargs):
    """
    For now, a 'plot' is a scatterplot.  At some point I will make 'plot'
    generic as with R.
    """
    device = get_current_graphics_device()
    ax = device.next_axes
    plot_options, kwargs = _skim_plot_options(**kwargs)

    if y is None:
        y = x
        x = np.arange(len(y))

    sample_size = len(x)

    if s is None:
        s = 20 / np.sqrt(sample_size)
    if sample_size < hexbin_threshold:
        ax.scatter(x, y, s=s, **kwargs)
    else:
        ax.hexbin(x, y, **kwargs)
    _set_plot_options(ax, **plot_options)

    device.draw_current_axes()

    return device


def hist(x, density: bool = False, edgecolor="black", color=".75", add=False,
         ax=None, **kwargs):
    """
    Plot a histogram of x.

    Args:
      x: The variable to be plotted.
      density: If True then the area of the histogram bars sums to 1.
    """
    if ax is None:
        _, ax = plt.subplots(1, 1, 1)

    plot_options, kwargs = _skim_plot_options(**kwargs)
    ax.hist(x[np.isfinite(x)], edgecolor=edgecolor, density=density,
            color=color, **kwargs)
    _set_plot_options(ax, **plot_options)

    return ax


def barplot(x, labels=None, zero=True, ax=None, **kwargs):
    """
    Make a horizonal bar plot.

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
    if ax is None:
        device = get_current_graphics_device()
        ax = device.next_axes
    else:
        device = None

    x = x[::-1]
    if labels is not None:
        labels = labels[::-1]

    if labels is None and isinstance(x, pd.Series):
        labels = [str(lab) for lab in x.index]
    bar_locations = np.arange(len(x))
    plot_options, kwargs = _skim_plot_options(**kwargs)

    if labels is None:
        labels = [""] * len(x)

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
    return device


def boxplot(x, labels=None, add=False, ax=None, **kwargs):
    if ax is None:
        device = get_current_graphics_device()
        if add:
            ax = device.current_axes
        else:
            ax = device.next_axes

    if labels is None and isinstance(x, pd.DataFrame):
        labels = x.columns

    ax.boxplot(x, vert=False)
    _set_plot_options(ax, **kwargs)

    return ax


def time_series_boxplot(curves, time=None, ylim=None, ax=None, **kwargs):
    """
    Plot side-by-side boxplots showing the evolution of a distribution over
    time.

    Args:
      curves: A matrix or data frame.  Rows represent different curves or Monte
        Carlo draws.  Columns represent time.

      time: A collectiond of timestamps corresponding to the column labels of
        'curves'.  If None then timestamps will be taken as the column labels
        of 'curves' (if curves is a DataFrame), or else they will be assigned
        the indices 1, 2, 3, ... .

      ylim: Limits on the vertical axis.

      ax:  The plt.Axes object on which to draw the plot.

      **kwargs:  Extra arguments passed to 'boxplot'.
    """
    plot_options, kwargs = _skim_plot_options(**kwargs)
    _set_plot_options(ax, **plot_options)

    if time is None:
        if isinstance(curves, pd.DataFrame):
            time = curves.columns
        else:
            time_dim = curves.shape[1]
            time = np.linspace(
                1, time_dim, num=time_dim).astype(int).astype(str)

    time = remove_common_prefix(
        remove_common_suffix(
            [str(x) for x in time]))
    ax.boxplot(curves, labels=time, **kwargs)
    return ax


def plot_ts(x, timestamps=None, ax=None, **kwargs):
    """ Plot a time series."""

    if ax is None:
        device = get_current_graphics_device()
        ax = device.next_axes

    if timestamps is None:
        if isinstance(x, pd.Series):
            timestamps = x.index

    ax.plot(x)
    _set_plot_options(ax, **kwargs)

    return ax


# ===========================================================================
# Custom plots
# ===========================================================================

def mosaic_plot(counts, ax=None, col_vname=None, row_vname=None):
    """
    Args:
      counts: A pd.DataFrame or equivalent, containing the contingency table
        describing the relationship between two categorical variables.
      ax:
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    else:
        fig = None

    assert isinstance(ax, plt.Axes)

    if isinstance(counts, np.ndarray):
        counts = pd.DataFrame(counts,
                              index=np.arange(counts.shape[0]).astype(str),
                              columns=np.arange(counts.shape[1]).astype(str))

    # The margininal distribution of the variable described by the rows,
    # obtained by summing over columns.
    row_margin = np.array(counts.sum(axis=1))
    nrow = len(row_margin)

    # The margininal distribution of the variable described by the columns,
    # obtained by summing over rows.
    col_margin = np.array(counts.sum(axis=0))
    ncol = len(col_margin)

    # The conditional distribution of the row variable, within each column.
    conditional = pd.DataFrame(counts / col_margin, index=counts.index,
                               columns=counts.columns)

    col_margin = col_margin / np.sum(col_margin)
    row_margin = row_margin / np.sum(row_margin)

    cum_col_margin = np.cumsum(col_margin)
    lower_col_margin = np.array([0] + cum_col_margin[:-1].tolist())
    column_positions = (lower_col_margin + cum_col_margin) / 2
    column_widths = cum_col_margin - lower_col_margin

    lower = np.zeros(ncol)
    for row_index in range(len(row_margin)):
        ax.bar(column_positions, conditional.iloc[row_index, :],
               width=column_widths, bottom=lower, edgecolor="gray",
               label=counts.index[row_index])
        lower += conditional.iloc[row_index, :]

    ax.set_xticks(column_positions)
    ax.set_xticklabels(counts.columns)

    cum_row_margin = np.cumsum(row_margin)
    row_low = np.array([0] + cum_row_margin[:-1].tolist())

    if np.min(row_margin) < .05:
        ax.set_yticks(np.linspace(0, 1, nrow))
    else:
        row_tick_locations = (cum_row_margin + row_low) / 2
        ax.set_yticks(row_tick_locations)
    ax.set_yticklabels(counts.index)

    if fig is not None:
        fig.show()

    return ax


def histabunch(data, min_continuous=12, max_levels=40, same_scale=False):
    nvars = data.shape[1]
    nr, nc = plot_grid_shape(nvars)
    _, ax = plt.subplots(nr, nc)

    def is_all_missing(y):
        return y.count() == 0

    def is_numeric(y):
        if isinstance(y, np.ndarray):
            y = pd.Series(y)
        return is_numeric_dtype(y) and len(y.value_counts()) >= min_continuous

    def is_categorical(y):
        return not is_numeric(y)

    def hist_categorical(y, vname, max_levels, ax):
        if isinstance(np.ndarray(y)):
            y = pd.Series(y)

        counts = y.value_counts()
        if counts.shape[0] > max_levels:
            counts = counts.sort_values(ascending=False)
            counts = counts[:max_levels]
        barplot(counts, ax=ax, main=vname)

    def hist_numeric(y, vname, ax):
        hist(y, ax=ax, main=vname)

    def plot_all_missing(vname, ax):
        pass

    vnames = data.columns
    plot_number = 0
    for row in range(nr):
        for col in range(nc):
            # Life would be better if we could reshape ax to ensure that it was
            # a matrix.  But alas.
            if len(ax.shape) == 2:
                ax_index = (row, col)
            else:
                ax_index = row + col
            if plot_number < nvars:
                vname = vnames[plot_number]
                y = data.loc[:, vname]

                if is_all_missing(y):
                    plot_all_missing(vname, ax[ax_index])
                elif is_numeric(y):
                    hist_numeric(y, vname, ax[ax_index])
                elif is_categorical(y):
                    hist_categorical(y, vname, max_levels, ax=ax[ax_index])
                plot_number += 1


def plot_many_ts(series, same_scale=True, ylim=None, gap=0, truth=None,
                 **kwargs):
    if len(series.shape) == 2:
        nseries = series.shape[1]
        nr, nc = plot_grid_shape(nseries)
    elif len(series.shape) == 3:
        nr, nc = series.shape[1], series.shape[2]
        nseries = nr * nc
    else:
        raise Exception("Wrong shape for 'series' argument to plot_many_ts.")

    gap *= .2
    fig, ax = plt.subplots(nr, nc)
    fig.subplots_adjust(hspace=gap, wspace=gap)

    series_number = 0
    if same_scale:
        ylim = data_range(series)

    if truth is not None:
        if isinstance(truth, Number):
            truth = np.ones((nr, nc)) * truth
        elif len(truth.shape) == 1:
            truth = np.concatenate(
                truth, np.zeros(nr * nc - nseries)).reshape((nr, nc))

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
                if len(series.shape) == 2:
                    the_series = series[:, series_number]
                else:
                    the_series = series[:, i, j]
                ax[i, j].plot(the_series)
                if truth is not None:
                    abline(ax=ax[i, j], h=truth[i, j])
                series_number += 1

    return fig, ax


def plot_dynamic_distribution(
        curves,
        timestamps=None,
        quantile_step=.02,
        xlab="Time",
        ylab="distribution",
        col="black",
        ax=None,
        **kwargs):
    """
    Plot a dynamic distribution represented by a collection of curves simulated
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

      **kwargs: Extra arguments passed to _skim_plot_options.
    """
    redraw = False
    if ax is None:
        device = get_current_graphics_device()
        ax = device.next_axes
        redraw = True

    plot_options, kwargs = _skim_plot_options(**kwargs)

    quantile_points = np.arange(0, 1, quantile_step)
    curve_quantiles = np.quantile(curves, q=quantile_points, axis=0)
    if timestamps is None:
        timestamps = np.arange(curves.shape[1])
    if len(timestamps) != curves.shape[1]:
        raise Exception("len(timestamps) must match curves.shape[1].")

    for i in range(int(np.floor(len(quantile_points) / 2))):
        lo = curve_quantiles[i, :]
        hi = curve_quantiles[-1-i, :]
        ax.fill_between(timestamps, lo, hi, color=col, edgecolor="none",
                        alpha=(i / len(quantile_points)))

    _set_plot_options(ax, **plot_options)

    if redraw:
        device = get_current_graphics_device()
        device.draw_current_axes()


def compare_dynamic_distributions(
        list_of_curves,
        timestamps,
        style="dynamic",
        xlab="Time",
        ylab="",
        frame_labels=None,
        main="",
        actuals=None,
        col_actuals=None,
        pch_actuals="o",
        cex_actuals=1,
        vertical_cuts=None,
        fig=None,
        **kwargs):
    """
    Produce a plot showing several stacked dynamic distributions over the same
    horizontal axis.

    Args:
      list.of.curves: A list of matrices, all having the same number of
         columns.  Each matrix represents a distribution of curves, with rows
         corresponding to individual curves, and columns to time points.
      timestamps: A vector of time stamps, with length matching the number of
        columns in each element of list.of.curves.
      style: Should the curves be represented using a dynamic distribution
        plot, or boxplots.  Boxplots are better for small numbers of time
        points.  Dynamic distribution plots are better for large numbers of
        time points.
      xlab:  Label for the horizontal axis.
      ylab:  Label for the (outer) vertical axis.
      frame.labels: Labels for the vertical axis of each subplot. The length
        must match the number of plot.
      main:  Main title for the plot.
      actuals: If non-NULL, actuals should be a numeric vector giving the
        actual "true" value at each time point.
      col_actuals:  Color to use for the actuals.
      cex_actuals:  Scale factor for actuals.
      vertical.cuts: If not None then this must be a vector of the same type as
        'timestamps' with length matching the number of plots.  A vertical line
        will be drawn at this location for each plot.  Entries with the value
        NaN or NaT signal that no vertical line should be drawn for that entry.
      kwargs: Extra arguments passed to PlotDynamicDistribution or
       TimeSeriesBoxplot.
    """
    style = unique_match(style, ["dynamic", "boxplot"])
    nplots = len(list_of_curves)
    ntimes = len(timestamps)
    for i in range(nplots):
        if list_of_curves[i].shape[1] != ntimes:
            raise Exception(f"Entry {i} in 'list_of_curves' did not have the "
                            f"right number of columns.  Expected {ntimes}, "
                            f"got {list_of_curves[i].shape[1]}.")
    if frame_labels is None:
        frame_labels = [str(i + 1) for i in range(nplots)]
    if not len(frame_labels) == nplots:
        raise Exception("frame labels do not match number of curves.")
    if fig is None:
        fig = plt.figure()

    ax = fig.subplots(nplots, 1, sharex=True)
    if nplots == 1:
        # If there is only one plot, put ax in a list so we can "iterate" over
        # it without breaking the multiplot code.
        ax = [ax]
    for i in range(nplots):
        if style == "dynamic":
            plot_dynamic_distribution(list_of_curves[i],
                                      timestamps=timestamps,
                                      ylab=frame_labels[i],
                                      ax=ax[i],
                                      **kwargs)
        elif style == "boxplot":
            time_series_boxplot(list_of_curves[i],
                                timestamps=timestamps,
                                ylab=frame_labels[i],
                                ax=ax[i],
                                **kwargs)
        else:
            raise Exception(f"Unrecognized style {style}")

        if actuals is not None:
            ax[i].scatter(timestamps, actuals,
                          s=cex_actuals * 100 / np.sqrt(len(actuals)))

        if vertical_cuts is not None:
            if vertical_cuts[i] == vertical_cuts[i]:
                ax[i].axvline(vertical_cuts[i])

    if main:
        fig.suptitle(main)
    return fig


def hosmer_lemeshow_plot(actual, predicted, ax=None, **kwargs):
    """
    Construct a Hosmer Lemeshow plot on the supplied axes.  A Hosmer Lemeshow
    plot partitions the predicted values into groups, and compares the midpoint
    of each group with the observed proportion in the group.

    Args:
      actual: The actual set of 0's and 1's being modeled.
      predicted: The predicted probabilities coresponding to 'actual'.
      ax: The pyplot axes on which to draw the plot, or None.  If None the
        figure will be shown on function exit.
      **kwargs: Passed to plt.subplots in the event that ax is None.

    Return:
      ax: The axes object containing the plot.
      group_means: The pd.Series containing the group means.  The series is
        indexed by pd.Interval objects indicating the interval over which the
        means are averaged.
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = None
    group_means = pd.DataFrame({"pred": predicted, "actual": actual}).groupby(
        pd.qcut(predicted, 10))["actual"].mean()
    bar_locations = group_means.index.categories.mid.values

    lower = np.array([x.left for x in group_means.index.values])
    upper = np.array([x.right for x in group_means.index.values])
    bar_widths = 1.0 * (upper - lower)

    plot_options, kwargs = _skim_plot_options(**kwargs)

    ax.bar(bar_locations, group_means, width=bar_widths, edgecolor="black", **kwargs)
    # ax.set_xlim(0, 1)
    # ax.set_ylim(0, 1)

    # labels = [str(lab) for lab in group_means.index.values]
    # ax.set_yticklabels(labels)
    ax.set_ylabel("Observed Proportions")

    # xticks = pretty_plot_ticks(np.min(predicted), np.max(predicted), 5)
    # ax.set_xticks(xticks)
    ax.set_xlabel("Predicted Proportions")

    _set_plot_options(ax, **plot_options)

    abline(a=0, b=1, ax=ax, color="black")

    if fig is not None:
        fig.show()

    return ax, group_means


def lines_gaussian_kde(kde, ax=None, **kwargs):
    """
    Add a kernel density estimate to the plot.
    """
    if ax is None:
        device = get_current_graphics_device()
        ax = device.current_axes
    xlim = ax.get_xlim()
    x = np.linspace(xlim[0], xlim[1])
    y = kde.pdf(x)
    ax.plot(x, y, **kwargs)


def lty(style):
    """
    Python linestyle characters from R 'lty' (linetype).

    Args:
      style:  An int or string.

    """
    style_names = ["solid", "dashed", "dotted", "dotdash",
                   "longdash", "twodash"]

    mappings = {
        "solid": "-",
        "dashed": "--",
        "dotted": ":",
        "dotdash": "-.",
        "longdash": (5, (8, 1, 8, 1)),
        "twodash": (5, (6, 1, 3, 1))
    }

    if isinstance(style, str):
        if style == "dashdot":
            style = "dotdash"
        style = unique_match(style, style_names)
        return mappings[style]

    elif isinstance(style, Number):
        style_number = style % len(style_names)
        return mappings[style_names[style_number]]
