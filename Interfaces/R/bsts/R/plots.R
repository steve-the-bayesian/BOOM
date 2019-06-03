# Copyright 2018 Google LLC. All Rights Reserved.
#
# This library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with this library; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA

qqdist <- function(draws, ...) {
  ## The distribution of a QQ plot for a set of noisy observations thought to be
  ## normally distributed.
  ##
  ## Args:
  ##   draws: A matrix of Monte Carlo draws.  Rows correspond to draws.  Columns
  ##     to noisy observations.
  ##   ...: Extra arguments passed to PlotDynamicDistribution.
  ##
  ## Effects:
  ##   A dynamic distribution plot is added to the graphics device, showing the
  ##   posterior distribution of the observations, sorted by their posterior
  ##   mean.  The plot also contains a reference line showing the expected
  ##   values of the points under perfect normality, and blue dots based on the
  ##   posterior means of the noisy observations.

  ## Step 1: draw the dynamic distribution plot.
  post.mean <- colMeans(draws)
  sample.size <- length(post.mean)
  draws <- draws[, order(post.mean)]
  expected <- qnorm(ppoints(sample.size))
  PlotDynamicDistribution(draws, timestamps = expected,
    xlab = "Quantiles of Standard Normal", ylab = "Distribution",
    ...)

  ## Step 2: Add the reference line.
  probs <- c(.25, .75)
  x <- qnorm(probs)
  y <- quantile(post.mean, probs)
  abline(lsfit(x, y), col = "green")

  ## Step3:  Add the points from the posterior means.
  points(expected, sort(post.mean), pch = 20, col = "blue")

  return(invisible(NULL))
}

AcfDist <- function(draws, lag.max = NULL,
                    xlab = "Lag", ylab = "Autocorrelation", ...) {
  ## Plot the posterior distribution of the autocorrelation function for 'draws'.
  ##
  ## Args:
  ##   draws: A matrix representing the posterior distribution of a time series.
  ##     Each row is a Monte Carlo draw, and each column is a time point.
  ##   lag.max:  The number of lags in the ACF.  Passed directly to 'acf'.
  ##   xlab:  Label for the horizontal axis.
  ##   ylab:  Label for the vertical axis.
  ##   ...:  Extra arguments passed to 'boxplot'.x
  ##
  ## Details:
  ##   A sequence of boxplots is plotted, each giving the marginal posterior
  ##   distribution of the ACF at a specific lag.
  ##
  ## Returns:
  ##   invisible(NULL).
  dist <- t(apply(draws, 1, function(x) {
    return(acf(x, plot = FALSE, lag.max = lag.max)$acf)
  }))
  lag.names <- as.character(seq(from = 0, len = ncol(dist), by = 1))
  boxplot(dist, xlab = xlab, ylab = ylab, names = lag.names, ...)
  abline(h = 0)
  return(invisible(NULL))
}

DayPlot <- function(y, colors = NULL, ylab = NULL, ...) {
  ## Plot the time series of each day of the week, all on the same plot (so
  ## seven lines on the same plot, for Sunday, Monday, ...)
  ## Args:
  ##   y: A zoo object containing daily data.  The index of y must be either
  ##     Date or POSIXt.
  ##   colors:  A vector of colors to use for the lines.
  ##   ylab:  The label for the vertical axis.
  ##
  ## Effects:
  ##   Plots seven time series on the same set of axes, showing the time series
  ##   of Mondays, Tuesdays, etc.  This is obviously only useful for data
  ##   observed at daily or finer time scales.
  return(MonthPlot(y, seasonal.identifier = base::weekdays,
    colors = colors, ylab = ylab, ...))
}

YearPlot <- function(y, colors = NULL, ylab = NULL, ylim = NULL, legend = TRUE, ...) {
  ## Overlay the time plot for each year of the time series on the same axis.
  ##
  ## Args:
  ##   y: A zoo object to be plotted.  The index of y must have class Date or
  ##     POSIXt.
  ##   colors:  A vector of colors to use for the lines.
  ##   ylab:  The label for the vertical axis.
  ##   ylim:  Limits for the vertical axis.
  ##   legend:  Logical.  If TRUE then a legend is added to the plot.
  ##
  ## Effects:
  ##   A plot is added to the current graphics device.  Each year of y is shown
  ##   as a separate line on the plot.  This plot is most effective when there
  ##   are a modest number of years (say 20 or less) of time series data on the
  ##   monthly, weekly, or daily scale.
  if (is.null(ylab)) {
    ylab <- deparse(substitute(y))
  }
  if (is.ts(y)) {
    y <- as.zoo(y)
  }
  stopifnot(is.zoo(y))
  if (inherits(index(y), "yearmon")) {
    index(y) <- YearMonToPOSIX(index(y))
  }
  stopifnot(inherits(index(y), c("Date", "POSIXt")))
  if (inherits(index(y), "Date")) {
    index(y) <- DateToPOSIX(index(y))
  }
  
  years <- sort(unique(sapply(strsplit(as.character(index(y)), "-"),
    function(x) as.numeric(x[1]))))
  
  ts.list <- list()
  for (i in 1:length(years)) {
    year <- years[i]
    begin <- as.POSIXct(paste0(year, "-01-01"))
    end <- as.POSIXct(paste0(year, "-12-31"))
    yearly.series <- window(y, start = begin, end = end)

    ##### Daily data assumed here.
    idx <- as.POSIXlt(as.character(index(yearly.series)))
    ## POSIXlt stores the year relative to 1900.
    idx$year <- years[1] - 1900
    index(yearly.series) <- idx
    ts.list[[i]] <- yearly.series
  }
  if (is.null(colors)) {
    colors <- 1:length(ts.list)
  }
  stopifnot(length(colors) >= length(ts.list))
  names(ts.list) <- as.character(years)
  time.index <- seq(DateToPOSIX(as.Date(paste0(years[1], "-01-01"))),
    length = 365,
    by = "day")
  if(is.null(ylim)) {
    ylim <- range(y, na.rm = TRUE)
  }
  plot(ts.list[[1]], ylim = ylim, xlim = range(time.index, na.rm = TRUE),
    ylab = ylab, ...)
  for (i in 2:length(years)) {
    lines(ts.list[[i]], lty = i, col = colors[i], ...)
  }
  if (legend) {
    legend("topleft",
      legend = as.character(years),
      lty = 1:length(years),
      col = colors[1:length(years)],
      bg = "white")
  }
  return(invisible(NULL))
}

MonthPlot <- function(y, seasonal.identifier = months, colors = NULL, ylab = NULL, ...) {
  ## Plot the time series of each day of the week, all on the same plot (so
  ## seven lines on the same plot, for Sunday, Monday, ...)
  ## Args:
  ##   y:  A ts or zoo object containing seasonal data.
  ##   seasonal.identifier: a function that takes a vector of timestamps as an
  ##     argument, and returns the name of the season containing the timestamp.
  ##     See ?weekdays for other options.
  ##   colors:  A vector of colors to use for the lines.
  ##   ylab:  Label for the vertical axis.
  ##   ...: Other arguments passed to plot and lines.
  if (is.null(ylab)) {
    ylab <- deparse(substitute(y))
  }
  if (is.ts(y)) {
    y <- as.zoo(y)
  }
  stopifnot(is.zoo(y))
  stopifnot(inherits(index(y), c("Date", "POSIXt", "yearmon")))
  if (inherits(index(y), "yearmon")) {
    index(y) <- YearMonToPOSIX(index(y))
  } else if (inherits(index(y), "Date")) {
    index(y) <- DateToPOSIX(index(y))
  }

  season.names <- unique(seasonal.identifier(index(y)))
  if (is.null(colors)) {
    colors <- 1:length(season.names)
  }
  stopifnot(length(colors) >= length(season.names))
  
  for (i in 1:length(season.names)) {
    season <- season.names[i]
    index <- as.character(seasonal.identifier(index(y))) == season
    if (i == 1) {
      plot(y[index], ylim = range(y, na.rm = TRUE), col = colors[i], ylab = ylab, ...)
    } else {
      lines(y[index], col = i, ...)
    }
  }
  abline(v = seq(from = min(index(y)), to = max(index(y)), by = "year"), lty = 3)
  legend("topleft", col = colors[1:length(season.names)],
    legend = season.names, bg = "white", lty = 1)
  return(invisible(NULL))
}

plot.bsts <- function(x,
                      y = c("state", "components", "residuals", "coefficients",
                          "prediction.errors", "forecast.distribution",
                          "predictors", "size",
                          "dynamic", "seasonal", "monthly", "help"),
                      ...) {
  ## S3 method for plotting bsts objects.
  ## Args:
  ##   x: An object of class 'bsts'.
  ##   y: character string indicating the aspect of the model that
  ##     should be plotted.  Partial matching is allowed,
  ##     so 'y = "res"' will produce a plot of the residuals.
  ## Returns:
  ##   This function is called for its side effect, which is to
  ##   produce a plot on the current graphics device.
  y <- match.arg(y)
  if (y == "state") {
    PlotBstsState(x, ...)
  } else if (y == "components") {
    PlotBstsComponents(x, ...)
  } else if (y == "residuals") {
    PlotBstsResiduals(x, ...)
  } else if (y == "coefficients") {
    PlotBstsCoefficients(x, ...)
  } else if (y == "prediction.errors") {
    PlotBstsPredictionErrors(x, ...)
  } else if (y == "forecast.distribution") {
    PlotBstsForecastDistribution(x, ...)
  } else if (y == "predictors") {
    PlotBstsPredictors(x, ...)
  } else if (y == "size") {
    PlotBstsSize(x, ...)
  } else if (y == "dynamic") {
    PlotDynamicRegression(x, ...)
  } else if (y == "seasonal") {
    PlotSeasonalEffect(x, ...)
  } else if (y == "monthly") {
    PlotMonthlyAnnualCycle(x, ...)
  } else if (y == "help") {
    help("plot.bsts", package = "bsts", help_type = "html")
  }
}
###----------------------------------------------------------------------
PlotBstsPredictors <- function(bsts.object,
                               burn = SuggestBurn(.1, bsts.object),
                               inclusion.threshold = .10,
                               ylim = NULL,
                               flip.signs = TRUE,
                               show.legend = TRUE,
                               grayscale = TRUE,
                               short.names = TRUE,
                               ...) {
  ## Plots the time series of predictors with high inclusion
  ## probabilities.
  ## Args:
  ##   bsts.object:  A bsts object containing a regression component.
  ##   burn:  The number of MCMC iterations to discard as burn-in.
  ##   inclusion.threshold: An inclusion probability that coefficients
  ##     must exceed in order to be displayed.
  ##   ylim:  Limits on the vertical axis.
  ##   flip.signs: If true then a predictor with a negative sign will
  ##     be flipped before being plotted, to better align visually
  ##     with the target series.
  ##   ...:  Extra arguments passed to either 'plot' or 'plot.zoo'.
  ## Returns:
  ##   Invisible NULL.
  stopifnot(inherits(bsts.object, "bsts"))
  if (HasDuplicateTimestamps(bsts.object)) {
    stop("The bsts predictor plot is does not support multiple time stamps.")
  }
  beta <- bsts.object$coefficients
  if (burn > 0) {
    beta <- beta[-(1:burn), , drop = FALSE]
  }
  inclusion.probabilities <- colMeans(beta != 0)
  keep <- inclusion.probabilities > inclusion.threshold
  if (any(keep)) {

    predictors <- bsts.object$predictors[, keep, drop = FALSE]
    predictors <- scale(predictors)
    if (flip.signs) {
      compute.positive.prob <- function(x) {
        x <- x[x != 0]
        if (length(x) == 0) {
          return(0)
        }
        return(mean(x > 0))
      }
      positive.prob <- apply(beta[, keep, drop = FALSE], 2,
                             compute.positive.prob)
      signs <- ifelse(positive.prob > .5, 1, -1)
      predictors <- scale(predictors, scale = signs)
    }

    inclusion.probabilities <- inclusion.probabilities[keep]
    number.of.predictors <- ncol(predictors)
    original <- scale(bsts.object$original.series)
    if (is.null(ylim)) {
      ylim <- range(predictors, original, na.rm = TRUE)
    }
    index <- rev(order(inclusion.probabilities))
    predictors <- predictors[, index]
    inclusion.probabilities <- inclusion.probabilities[index]
    predictor.names <- colnames(predictors)
    if (short.names) {
      predictor.names <- Shorten(predictor.names)
    }

    if (grayscale) {
      line.colors <- gray(1 - inclusion.probabilities)
    } else {
      line.colors <- rep("black", number.of.predictors)
    }
    times <- bsts.object$timestamp.info$timestamps
    if (number.of.predictors == 1) {
      plot(times, predictors, type = "l", lty = 1, col = line.colors,
           ylim = ylim, xlab = "", ylab = "Scaled Value", ...)
    } else {
      plot(times, predictors[, 1], type = "n", ylim = ylim, xlab = "",
           ylab = "Scaled Value", ...)
      for (i in 1:number.of.predictors) {
        lines(times, predictors[, i], lty = i, col = line.colors[i], ...)
      }
    }
    lines(zoo(scale(bsts.object$original.series), times),
          col = "blue",
          lwd = 3)
    if (show.legend) {
      legend.text <- paste(round(inclusion.probabilities, 2), predictor.names)
      legend("topright", legend = legend.text, lty = 1:number.of.predictors,
             col = line.colors, bg = "white")
    }
  } else {
    plot(0, 0, type = "n",
         main = "No predictors above the inclusion threshold.", ...)
  }
  return(invisible(NULL))
}

###----------------------------------------------------------------------
PlotBstsCoefficients <- function(bsts.object,
                                 burn = SuggestBurn(.1, bsts.object),
                                 inclusion.threshold = 0,
                                 number.of.variables = NULL,
                                 ...) {
  ## Creates a plot of the regression coefficients in the bsts.object.
  ## This is a wrapper for plot.lm.spike from the BoomSpikeSlab package.
  ## Args:
  ##   bsts.object:  An object of class 'bsts'
  ##   burn: The number of MCMC iterations to discard as burn-in.
  ##   inclusion.threshold: An inclusion probability that coefficients
  ##     must exceed in order to be displayed.
  ##   number.of.variables: If non-NULL this specifies the number of
  ##     coefficients to plot, taking precedence over
  ##     inclusion.threshold.
  ## Returns:
  ##   Invisibly returns a list with the following elements:
  ##   barplot: The midpoints of each bar, which is useful for adding
  ##     to the plot
  ##   inclusion.prob: The marginal inclusion probabilities of each
  ##     variable, ordered smallest to largest (the same ordering as
  ##     the plot).
  ##   positive.prob: The probability that each variable has a
  ##     positive coefficient, in the same order as inclusion.prob.
  ##   permutation: The permutation of beta that puts the coefficients
  ##     in the same order as positive.prob and inclusion.prob.  That
  ##     is: beta[, permutation] will have the most significant
  ##     coefficients in the right hand columns.
  stopifnot(inherits(bsts.object, "bsts"))
  if (is.null(bsts.object$coefficients)) {
    stop("No coefficients to plot in PlotBstsCoefficients.")
  }
  return(invisible(
      PlotMarginalInclusionProbabilities(
          bsts.object$coefficients,
          burn = burn,
          inclusion.threshold = inclusion.threshold,
          number.of.variables = number.of.variables,
          ...)))
}
###----------------------------------------------------------------------
PlotBstsSize <- function(bsts.object,
                         burn = SuggestBurn(.1, bsts.object),
                         style = c("histogram", "ts"),
                         ...) {
  ## Plots the distribution of the number of variables in the bsts model.
  ## Args:
  ##   bsts.object:  An object of class 'bsts' to plot.
  ##   burn: The number of MCMC iterations to discard as burn-in.
  ##   style:  The desired plot style.
  ##   ...:  Extra arguments passed to lower level plotting functions.
  ## Returns:
  ##   Nothing interesting.  Draws a plot on the current graphics device.
  beta <- bsts.object$coefficients
  if (is.null(beta)) {
    stop("The model has no coefficients")
  }
  if (burn > 0) {
    beta <- beta[-(1:burn), , drop = FALSE]
  }
  size <- rowSums(beta != 0)
  style <- match.arg(style)
  if (style == "ts") {
    plot.ts(size, ...)
  } else if (style == "histogram") {
    hist(size, ...)
  }
  return(invisible(NULL))
}

###----------------------------------------------------------------------
PlotBstsComponents <- function(bsts.object,
                               burn = SuggestBurn(.1, bsts.object),
                               time = NULL,
                               same.scale = TRUE,
                               layout = c("square", "horizontal", "vertical"),
                               style = c("dynamic", "boxplot"),
                               ylim = NULL,
                               components = 1:length(bsts.object$state.specification),
                               ...) {
  ## Plots the posterior distribution of each state model's contributions to the
  ## mean of the time series.
  ##
  ## Args:
  ##   bsts.object: An object of class 'bsts'.
  ##   burn: The number of MCMC iterations to be discarded as burn-in.
  ##   time: An optional vector of values to plot on the time axis.
  ##   same.scale: Logical.  If TRUE then all plots will share a common scale
  ##     for the vertical axis.  Otherwise the veritcal scales for each plot
  ##     will be determined independently.
  ##   layout: A text string indicating whether the state components plots
  ##     should be laid out in a square (maximizing plot area), vertically, or
  ##     horizontally.
  ##   style: Either "dynamic", for dynamic distribution plots, or "boxplot",
  ##     for box plots.  Partial matching is allowed, so "dyn" or "box" would
  ##     work, for example.
  ##   ylim:  Limits on the vertical axis.
  ##   components: A numeric index vector indicating which state components to
  ##     plot.
  ##   ...: Extra arguments passed to PlotDynamicDistribution.
  ## 
  ## Returns:
  ##   Produces a plot on the current graphics device.  Returns invisible NULL.
  stopifnot(is.numeric(components),
    length(components) >= 1,
    all(components == round(components)),
    min(components) >= 1,
    max(components) <= length(bsts.object$state.specification))
  components <- unique(components)
  
  state.specification <- bsts.object$state.specification[components]
  number.of.components <- length(state.specification)
  if (bsts.object$has.regression) {
    number.of.components <- number.of.components + 1
  }
  
  if (is.null(time)) {
    time <- bsts.object$timestamp.info$regular.timestamps
  }

  layout <- match.arg(layout)
  if (layout == "square") {
    num.rows <- floor(sqrt(number.of.components))
    num.cols <- ceiling(number.of.components / num.rows)
  } else if (layout == "vertical") {
    num.rows <- number.of.components
    num.cols <- 1
  } else if (layout == "horizontal") {
    num.rows <- 1
    num.cols <- number.of.components
  }

  screen.matrix <- .ScreenMatrix(num.rows, num.cols, side.margin = 0)
  close.screen(all.screens = TRUE)
  original.par <- par(mfrow = c(1,1))
  par(mfrow = c(num.rows, num.cols))
  cex <- par("cex")
  par(mfrow = c(1,1), cex= cex)
  
  screen.numbers <- split.screen(screen.matrix)
  on.exit({close.screen(screen.numbers); par(original.par)})

  if (is.null(ylim) && same.scale) {
    if (burn > 0) {
      ylim <- range(bsts.object$state.contributions[-(1:burn), , ])
    } else {
      ylim <- range(bsts.object$state.contributions)
    }
  }
  if (is.null(burn)) {
    burn <- 0
  }
  for (i in 1:length(components)) {
    screen(screen.numbers[i])
    ## Each state component has a generic plot function that knows how to make
    ## the right plot.
    plot(state.specification[[i]], bsts.object, burn = burn,
      time = time, style = style, ylim = ylim, ...)
  }
  if (bsts.object$has.regression) {
    screen(screen.numbers[length(components) + 1])
    .PlotRegressionComponent(bsts.object, burn = burn, time = time,
      style = style, ylim = ylim, ...)
  }
  return(invisible(NULL))
}

.PlotRegressionComponent <- function(bsts.object,
                                     burn = NULL,
                                     time = NULL,
                                     style = c("dynamic", "boxplot"),
                                     ylim = NULL,
                                     ...) {
  ## Plot the regression component of a state space model.
  ##
  ## Args:
  ##   bsts.object:  The object from which state.contribution was extracted.
  ##   burn:  The number of observations to discard as burn-in.
  ##   time:  A vector of time stamps to plot against.
  ##   style:  The style of plot desired.
  ##   ylim:  Limits for the vertical axis.
  ##   ...: Extra arguments passed to TimeSeriesBoxplot or
  ##     PlotDynamicDistribution.
  stopifnot(inherits(bsts.object, "bsts"), bsts.object$has.regression)
  state.contribution <- bsts.object$state.contributions[, "regression", ]
  style <- match.arg(style)
  .PlotStateContribution(state.contribution, bsts.object, burn, time, style,
    ylim, ...)
  title(main = "Regression")
  return(invisible(NULL))
}

.PlotStateContribution <- function(state.contribution,
                                   bsts.object,
                                   burn = NULL,
                                   time = NULL,
                                   style = c("dynamic", "boxplot"),
                                   ylim = NULL,
                                   ...) {
  ## Create a dynamic distribution plot or time series boxplot showing the
  ## contribution of a state component to the mean of the time series being
  ## modeled.
  ##
  ## Args:
  ##   state.contribution: A matrix of MCMC draws.  Each row is a draw.  Each
  ##     column is a time point.
  ##   bsts.object:  The object from which state.contribution was extracted.
  ##   burn:  The number of observations to discard as burn-in.
  ##   time:  A vector of time stamps to plot against.
  ##   style:  The style of plot desired.
  ##   ylim:  Limits for the vertical axis.
  ##   ...: Extra arguments passed to TimeSeriesBoxplot or
  ##     PlotDynamicDistribution.
  if (!is.matrix(state.contribution)) {
    state <- matrix(state.contribution, ncol = 1)
  }
  if (is.null(burn)) {
    burn <- 0
  }
  if (burn > 0) {
    state.contribution <- state.contribution[-(1:burn), ]
  }
  if (is.null(time)) {
    time <- bsts.object$timestamp.info$regular.timestamps
  }
  if (is.null(ylim)) {
    ylim <- range(state.contribution)
  }
  style <- match.arg(style)
  if (style == "boxplot") {
    TimeSeriesBoxplot(state.contribution, time = time, ylim = ylim, ...)
  } else {
    PlotDynamicDistribution(state.contribution, timestamps = time, ylim = ylim, ...)
  }

}

plot.StateModel <- function(x,
                            bsts.object,
                            burn = NULL,
                            time = NULL,
                            style = c("dynamic", "boxplot"),
                            ylim = NULL,
                            ...) {
  ## The default plotting method for a StateModel is to find the column of
  ## bsts.object$state.contributions corresponding to state.specification$name
  ## and plot it using either PlotDynamicRegression or TimeSeriesBoxplot.
  ##
  ## Args:
  ##   x:  An object inheriting from StateModel.
  ##   bsts.object: A bsts model that includes state.specification in its state
  ##     specification.
  ##   burn:  The number of MCMC iterations to burn.
  ##   time: An optional vector of values to plot on the time axis.
  ##   style: Either "dynamic", for dynamic distribution plots, or "boxplot",
  ##     for box plots.  Partial matching is allowed, so "dyn" or "box" would
  ##     work, for example.
  ##   ylim:  Limits on the vertical axis.
  ##   ...: Extra arguments passed to PlotDynamicDistribution or
  ##     TimeSeriesBoxplot.
  ##
  ## Returns:
  ##   Draws a plot on the current plotting device.
  state.specification <- x
  stopifnot(inherits(state.specification, "StateModel"))
  stopifnot(inherits(bsts.object, "bsts"))
  if (is.null(.FindStateSpecification(state.specification, bsts.object))) {
    stop("The state specification is not part of the bsts object.")
  }
  style <- match.arg(style)
  state <- bsts.object$state.contributions[, state.specification$name, ]
  if (is.null(state)) {
    stop("Could not find state contributions for ", state.specification$name)
  }
  .PlotStateContribution(state, bsts.object, burn, time, style, ylim, ...)
  title(main = state.specification$name)
  return(invisible(NULL))
}
  
###----------------------------------------------------------------------
PlotBstsState <- function(bsts.object, burn = SuggestBurn(.1, bsts.object),
                          time, show.actuals = TRUE,
                          style = c("dynamic", "boxplot"),
                          scale = c("linear", "mean"),
                          ylim = NULL,
                          ...) {
  ## Plots the posterior distribution of the mean of the training
  ## data, as determined by the state.
  ## Args:
  ##   bsts.object:  An object of class 'bsts'.
  ##   burn: The number of MCMC iterations to be discarded as burn-in.
  ##   time: An optional vector of values to plot on the time axis.
  ##   show.actuals: If TRUE then the original values from the series
  ##     will be added to the plot.
  ##   style: Either "dynamic", for dynamic distribution plots, or
  ##     "boxplot", for box plots.  Partial matching is allowed, so
  ##     "dyn" or "box" would work, for example.
  ##   scale: If the error family is logit or Poisson then setting
  ##     scale to "mean" will pass the model state through the logit
  ##     or exponential link functions before plotting.
  ##   ...: Extra arguments passed to PlotDynamicDistribution.
  ## Returns:
  ##   This function is called for its side effect, which is to
  ##   produce a plot on the current graphics device.
  stopifnot(inherits(bsts.object, "bsts"))
  style <- match.arg(style)
  scale <- match.arg(scale)
  if (missing(time)) {
    time <- bsts.object$timestamp.info$regular.timestamps
  }
  state <- bsts.object$state.contributions
  if (burn > 0) {
    state <- state[-(1:burn), , , drop = FALSE]
  }
  state <- rowSums(aperm(state, c(1, 3, 2)), dims = 2)
  if (scale == "mean") {
    if (bsts.object$family == "logit") {
      state <- t(t(plogis(state)) * bsts.object$trials)
    } else if (bsts.object$family == "poisson") {
      state <- t(t(exp(state)) * bsts.object$exposure)
    }
  } else {
    ## If the plot is not on the scale of the data then don't show the actual
    ## data values.
    if (bsts.object$family %in% c("poisson", "logit")) {
      show.actuals <- FALSE
    }
  }

  if (is.null(ylim)) {
    if (show.actuals) {
      ylim <- range(state, bsts.object$original.series, na.rm = TRUE)
    } else {
      ylim <- range(state, na.rm = TRUE)
    }
  }
  
  if (style == "boxplot") {
    TimeSeriesBoxplot(state, time = time, ylim = ylim, ...)
  } else {
    PlotDynamicDistribution(state, timestamps = time, ylim = ylim, ...)
  }
  if (show.actuals) {
    points(bsts.object$timestamp.info$timestamps,
           bsts.object$original.series,
           col = "blue",
           ...)
  }
  return(invisible(state))
}

###----------------------------------------------------------------------
PlotBstsPredictionErrors <- function(bsts.object,
                                     cutpoints = NULL,
                                     burn = SuggestBurn(.1, bsts.object),
                                     style = c("dynamic", "boxplot"),
                                     xlab = "Time",
                                     ylab = "",
                                     main = "",
                                     ...) {
  style <- match.arg(style)
  stopifnot(inherits(bsts.object, "bsts"))
  prediction.errors = bsts.prediction.errors(bsts.object,
                                             cutpoints = cutpoints,
                                             burn = burn)
  timestamps <- attributes(prediction.errors)$timestamps
  cutpoints <-  timestamps[attributes(prediction.errors)$cutpoints]
  CompareDynamicDistributions(
      prediction.errors,
      timestamps = timestamps,
      style = style,
      xlab = xlab,
      ylab = ylab,
      frame.labels = c(as.character(cutpoints), "in.sample"),
      main = main,
      actuals = NULL,
      vertical.cuts = c(cutpoints, NA),
      ...)
  return(invisible(prediction.errors))
}

###----------------------------------------------------------------------
PlotBstsForecastDistribution <- function(bsts.object,
                                         cutpoints = NULL,
                                         burn = SuggestBurn(.1, bsts.object),
                                         style = c("dynamic", "boxplot"),
                                         xlab = "Time",
                                         ylab = "",
                                         main = "",
                                         show.actuals = TRUE,
                                         col.actuals = "blue",
                                         ...) {
  ## Plots the posterior distribution of the one-step-ahead forecasts
  ## for a bsts model.  This is the distribution of p(y[t+1] | y[1:t],
  ## \theta) averaged over p(\theta | y[1:T]).
  ## Args:
  ##   bsts.object:  An object of class 'bsts'.
  ##   burn: The number of MCMC iterations to be discarded as burn-in.
  ##   style: Either "dynamic", for dynamic distribution plots, or
  ##     "boxplot", for box plots.  Partial matching is allowed, so
  ##     "dyn" or "box" would work, for example.
  ##   show.actuals: If TRUE then the original values from the series
  ##     will be added to the plot.
  ##   col.actuals: The color to use when plotting original values
  ##     from the time series being modeled.
  ##   ...: Extra arguments passed to TimeSeriesBoxplot,
  ##     PlotDynamicDistribution, and points.
  ##
  ## Returns:
  ##   invisible NULL
  stopifnot(inherits(bsts.object, "bsts"))
  style = match.arg(style)
  errors <- bsts.prediction.errors(bsts.object,
                                   cutpoints = cutpoints,
                                   burn = burn)
  forecast <- errors
  if (length(forecast) == 0) {
    stop("No forecast errors are available.")
  }
  for (i in 1:length(forecast)) {
    forecast[[i]] <-
        t(as.numeric(bsts.object$original.series) - t(forecast[[i]]))
  }
  timestamps <- attributes(errors)$timestamps
  cutpoints <- timestamps[attributes(errors)$cutpoints]
  if (show.actuals) {
    actuals <- bsts.object$original.series
  } else {
    actuals <- NULL
  }
  CompareDynamicDistributions(
      forecast,
      timestamps = timestamps,
      style = style,
      xlab = xlab,
      ylab = ylab,
      main = main,
      frame.labels = c(as.character(cutpoints), "in.sample"),
      actuals = actuals,
      vertical.cuts = c(cutpoints, NA),
      col.actuals = col.actuals,
      ...)
  return(invisible(forecast))
}

###----------------------------------------------------------------------
PlotBstsResiduals <- function(bsts.object, burn = SuggestBurn(.1, bsts.object),
                              time, style = c("dynamic", "boxplot"), means = TRUE,
                              ...) {
  ## Plots the posterior distribution of the residuals from the bsts
  ## model, after subtracting off the state effects (including
  ## regression effects).
  ## Args:
  ##   bsts.object:  An object of class 'bsts'.
  ##   burn: The number of MCMC iterations to be discarded as burn-in.
  ##   time: An optional vector of values to plot on the time axis.
  ##   means: If TRUE then the posterior mean of each residual is plotted as a
  ##     blue dot on top of the boxplot or the dynamic distribution plot.
  ##   style: Either "dynamic", for dynamic distribution plots, or
  ##     "boxplot", for box plots.  Partial matching is allowed, so
  ##     "dyn" or "box" would work, for example.
  ##   ...:  Extra arguments passed to PlotDynamicDistribution.
  ## Returns:
  ##   This function is called for its side effect, which is to
  ##   produce a plot on the current graphics device.
  stopifnot(inherits(bsts.object, "bsts"))
  if (HasDuplicateTimestamps(bsts.object)) {
    stop("The bsts residual plot does not support duplicate time stamps.")
  }
  if (bsts.object$family %in% c("poisson", "logit")) {
    stop("The bsts residual plot is only supported for continuous error",
         " families.")
  }
  style <- match.arg(style)
  if (missing(time)) {
    time <- bsts.object$timestamp.info$timestamps
  }
  residuals <- residuals(bsts.object)
  if (style == "dynamic") {
    PlotDynamicDistribution(residuals, timestamps = time, ...)
  } else {
    TimeSeriesBoxplot(residuals, time = time, ...)
  }

  if (means) {
    points(time, colMeans(residuals), pch = 20, col = "blue")
  }
  return(invisible(NULL))
}

###----------------------------------------------------------------------
PlotDynamicRegression <- function(
    bsts.object,
    burn = SuggestBurn(.1, bsts.object),
    time = NULL,
    same.scale = FALSE,
    style = c("dynamic", "boxplot"),
    layout = c("square", "horizontal", "vertical"),
    ylim = NULL,
    zero.width = 2,
    zero.color = "green",
    ...) {
  ## Plot the coefficients of a dynamic regression state component.
  ## Args:
  ##   bsts.object: The bsts object containing the dynamic regression
  ##     state component to be plotted.
  ##
  ##   burn: The number of MCMC iterations to be discarded as burn-in.
  ##   time: An optional vector of values to plot on the time axis.
  ##   layout: A text string indicating whether the state components
  ##     plots should be laid out in a square (maximizing plot area),
  ##     vertically, or horizontally.
  ##   style: Either "dynamic", for dynamic distribution plots, or
  ##     "boxplot", for box plots.  Partial matching is allowed, so
  ##     "dyn" or "box" would work, for example.
  ##   ...: Additional arguments passed to PlotDynamicDistribution or
  ##     TimeSeriesBoxplot.
  stopifnot(inherits(bsts.object, "bsts"))
  if (!("dynamic.regression.coefficients" %in% names(bsts.object))) {
    stop("The model object does not contain a dynamic regression component.")
  }
  style <- match.arg(style)
  if (is.null(time)) {
    time <- bsts.object$timestamp.info$regular.timestamps
  }
  beta <- bsts.object$dynamic.regression.coefficients
  ndraws <- dim(beta)[1]
  number.of.variables <- dim(beta)[2]
  stopifnot(length(time) == dim(beta)[3])

  if (burn > 0) {
    beta <- beta[-(1:burn), , , drop = FALSE]
  }
  if (is.null(ylim) && same.scale == TRUE) {
    ylim <- range(beta, na.rm = TRUE)
  }
  
  layout <- match.arg(layout)
  if (layout == "square") {
    num.rows <- floor(sqrt(number.of.variables))
    num.cols <- ceiling(number.of.variables / num.rows)
  } else if (layout == "vertical") {
    num.rows <- number.of.variables
    num.cols <- 1
  } else if (layout == "horizontal") {
    num.rows <- 1
    num.cols <- number.of.variables
  }
  original.par <- par(mfrow = c(num.rows, num.cols))
  on.exit(par(original.par))
  beta.names <- dimnames(beta)[[2]]
  need.ylim <- is.null(ylim)
  for (variable in 1:number.of.variables) {
    if (need.ylim) {
      ylim <- range(beta[, variable, ], na.rm = TRUE)
    }      
    if (style == "boxplot") {
      TimeSeriesBoxplot(beta[, variable, ],
        time = time, ylim = ylim, ...)
    } else if (style == "dynamic") {
      PlotDynamicDistribution(beta[, variable, ],
        timestamps = time, ylim = ylim, ...)
    }
    if (!is.null(zero.width) && !is.null(zero.color)) {
      abline(h = 0, lty = 3, lwd = zero.width, col = zero.color)
    }
    title(beta.names[variable])
  }
  return(invisible(NULL))
}
###----------------------------------------------------------------------
PlotHoliday <- function(holiday, model, show.raw.data = TRUE,
                        ylim = NULL, ...) {
  ## Plot the estimated holiday effect for the given holiday.
  ##
  ## Args:
  ##   holiday:  An object of class Holiday.
  ##   model: A bsts model object with a state specification that includes
  ##     'holiday' in a RegressionHolidayStateModel or
  ##     HierarchicalRegressionHolidayStateModel.
  ##   show.raw.data: Logical indicating if the raw data corresponding to the
  ##     holiday should be superimposed on the plot.  The 'raw data' are the
  ##     actual values of the target series, minus the value of the target
  ##     series the day before the holiday began, which is a (somewhat poor)
  ##     proxy for remaining state elements.  The raw data can appear
  ##     artificially noisy if there are other strong state effects such as a
  ##     day-of-week effect for holidays that don't always occur on the same day
  ##     of the week.
  ##   ylim:  Limits for the vertical axis.
  ##
  ## Effects:
  ##   A set of boxplots are drawn on the current graphics device showing the
  ##   posterior distribution of the impact of the holiday during each day of
  ##   its influence window.  
  date.ranges <- DateRange(holiday, model$timestamp.info$timestamps)
  raw.data<- list()
  for (i in 1:nrow(date.ranges)) {
    baseline <- as.numeric(model$original.series[date.ranges[i, 1] - 1])
    actuals <- as.numeric(window(model$original.series,
      start = date.ranges[i, 1],
      end = date.ranges[i, 2]))
    raw.data[[i]] <- actuals - baseline
  }

  holiday.effects <- .FindHolidayEffects(model, holiday$name)
  if (is.null(ylim)) {
    ylim <- range(holiday.effects, na.rm = TRUE)
    if (show.raw.data) {
      ylim <- range(ylim, raw.data, na.rm = TRUE)
    } 
  }

  boxplot(holiday.effects, ylim = ylim, ...)
  if (show.raw.data) {
    for (i in 1:length(raw.data)) {
      lines(raw.data[[i]], col = i, lty = i)
    }
  }
  return(invisible(NULL))
}

.FindHolidayEffects <- function(model, holiday.name) {
  ## Look through all the state specificiations and find all the ones that
  ## inherit from HolidayModel.
  ##
  ## Args:
  ##   model:  a bsts model to search.
  ##   holiday.name:  The name of a holiday expected to be part of 'model'.
  ##
  ## Returns:
  ##    If a holiday named 'holiday.name' was included in 'model' as part of a
  ##    RegressionHolidayStateModel or HierarchicalRegressionHolidayStateModel
  ##    then find the MCMC draws of its coefficients and return them as a
  ##    matrix. Otherwise raise an error.
  state.specification <- model$state.specification
  for (i in 1:length(state.specification)) {
    if (inherits(state.specification[[i]], "RegressionHolidayStateModel")) {
      holiday.names <- sapply(state.specification[[i]]$holidays, function(x)
        x$name)
      if (holiday.name %in% holiday.names) {
        return(model[[holiday.name]])
      }
    } else if (inherits(state.specification[[i]],
      "HierarchicalRegressionHolidayStateModel")) {
      return(model$holiday.coefficients[, holiday.name, ])
    } else if (inherits(state.specification[[i]], "RandomWalkHolidayStateModel")) {
      stop("A plot method for RandomWalkHolidayStateModel never got implemented.")
    }
  }
  stop(paste0("Could not find a holiday named ", holiday.name, "."))
  return(NULL)
}

      
