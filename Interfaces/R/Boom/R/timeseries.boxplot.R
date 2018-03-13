TimeSeriesBoxplot <- function(x, time, ylim = NULL, add = FALSE, ...) {
  ## Plots side by side boxplots against horizontal times series axis.
  ##
  ## Args:
  ##   x: A matrix, with rows representing a curve (or observations at a point
  ##     in time), and columns representing time points.
  ##   time: A vector of timestamps, with length equal to the number of columns
  ##     in x.  Timestamps can be numeric, Date, or POSIXt.
  ##   ylim:  Vertical axis limits.
  ##   add: Logical.  Should the boxplots be added to the current plot?
  ##   ...:  Extra arguments passed to 'plot' and 'boxplot'.
  stopifnot(inherits(time, "Date") ||
            inherits(time, "POSIXt") ||
            is.numeric(time),
            length(time) > 0,
            length(time) == ncol(x))
  if (is.null(ylim)) {
    ylim <- range(x)
  }
  if (!add) {
    plot(time, 1:length(time), type = "n", ylim = ylim, ...)
  }
  dt <- mean(diff(time))
  boxplot(x, add = TRUE, at = time, show.names = FALSE,
          boxwex = dt/2, ylim = ylim, ...)
}
