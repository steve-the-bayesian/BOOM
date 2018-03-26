CompareDynamicDistributions <- function(
    list.of.curves,
    timestamps,
    style = c("dynamic", "boxplot"),
    xlab = "Time",
    ylab = "",
    frame.labels = rep("", length(list.of.curves)),
    main = "",
    actuals = NULL,
    col.actuals = "blue",
    pch.actuals = 1,
    cex.actuals = 1,
    vertical.cuts = NULL,
    ...) {
  ## Produce a plot showing several stacked dynamic distributions over the same
  ## horizontal axis.
  ##
  ## Args:
  ##   list.of.curves: A list of matrices, all having the same number of
  ##     columns.  Each matrix represents a distribution of curves, with rows
  ##     corresponding to individual curves, and columns to time points.
  ##   timestamps: A vector of time stamps, with length matching the number of
  ##     columns in each element of list.of.curves.
  ##   style: Should the curves be represented using a dynamic distribution
  ##     plot, or boxplots.  Boxplots are better for small numbers of time
  ##     points.  Dynamic distribution plots are better for large numbers of
  ##     time points.
  ##   xlab:  Label for the horizontal axis.
  ##   ylab:  Label for the (outer) vertical axis.
  ##   frame.labels: Labels for the vertical axis of each subplot. The length
  ##     must match the number of plot.
  ##   main:  Main title for the plot.
  ##   actuals: If non-NULL, actuals should be a numeric vector giving the
  ##     actual "true" value at each time point.
  ##   col.actuals:  Color to use for the actuals.  See 'par'.
  ##   pch.actuals:  Plotting character(s) to use for the actuals.  See 'par'.
  ##   cex.actuals:  Scale factor for actuals.  See 'par'.
  ##   vertical.cuts: If non-NULL then this must be a vector of the same type as
  ##     'timestamps' with length matching the number of plots.  A vertical line
  ##     will be drawn at this location for each plot.  Entries with the value
  ##     NA signal that no vertical line should be drawn for that entry.
  ##   ...: Extra arguments passed to PlotDynamicDistribution or
  ##     TimeSeriesBoxplot.
  style <- match.arg(style)
  stopifnot(is.list(list.of.curves),
            length(list.of.curves) >= 1,
            all(sapply(list.of.curves, is.matrix)),
            length(unique(sapply(list.of.curves, ncol))) == 1)
  stopifnot(length(timestamps) == ncol(list.of.curves[[1]]))
  stopifnot(is.character(frame.labels),
            length(frame.labels) == length(list.of.curves))
  if (!is.null(actuals)) {
    stopifnot(is.numeric(actuals),
              length(actuals) == length(timestamps))
  }
  if (!is.null(vertical.cuts)) {
    stopifnot(length(vertical.cuts) == length(list.of.curves))
  }
  nplots <- length(list.of.curves)
  if (nplots > 1) {
    ylab.space <- if (ylab != "") 4.1 else 2
    original.par <- par(mfrow = c(nplots, 1),
                        mar = c(0, 4.1, 0, 2.1),
                        oma = c(5.1, ylab.space, 4.1, 2))
    on.exit(par(original.par))
  }
  for (i in seq_along(list.of.curves)) {
    if (style == "dynamic") {
      PlotDynamicDistribution(list.of.curves[[i]],
                              timestamps = timestamps,
                              axes = FALSE,
                              ylab = frame.labels[i],
                              ...)
    } else if (style == "boxplot") {
      TimeSeriesBoxplot(list.of.curves[[i]],
                        axes = FALSE,
                        time = timestamps,
                        ylab = frame.labels[i],
                        ...)
      box()
    }

    if (IsOdd(i)) {
      axis(2)
    } else {
      axis(4)
    }
    if (!is.null(vertical.cuts) && !is.na(vertical.cuts[i])) {
      abline(v = vertical.cuts[i], lwd = 2)
    }
    if (!is.null(actuals)) {
      points(timestamps, actuals, col = col.actuals, pch = pch.actuals,
             cex = cex.actuals)
    }
  }

  if (inherits(timestamps, "Date")) {
    axis.Date(1, timestamps, xpd = NA)
  } else if (inherits(timestamps, "POSIXt")) {
    axis.POSIXct(1, as.POSIXct(timestamps), xpd = NA)
  } else {
    axis(1, xpd = NA)
  }

  title(main = main, xlab = xlab, ylab = ylab, outer = TRUE)
  return(invisible(NULL))
}
