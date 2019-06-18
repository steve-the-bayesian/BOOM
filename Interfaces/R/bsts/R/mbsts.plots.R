# Copyright 2019 Steven L. Scott. All Rights Reserved.
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

plot.mbsts <- function(x,
                       y = c("means", "help"),
                       ...) {
  ## S3 method for plotting an mbsts object.
  ## Args:
  ##   x: The ojbect to plot.
  ##   y: A string giving the type of plot desired.
  ##   ...: Named arguments passed to the specific functions implementing the
  ##     plots.  See the plot.mbsts source for a list of these functions.
  y <- match.arg(y)
  if (y == "means") {
    PlotMbstsSeriesMeans(x, ...)
  } else if (y == "help") {
    help("plot.mbsts", package = "bsts", help_type = "html")
  } 
}

###---------------------------------------------------------------------------
PlotMbstsSeriesMeans <- function(mbsts.object,
                                 series.id = NULL,
                                 same.scale = TRUE,
                                 burn = SuggestBurn(.1, mbsts.object),
                                 time, 
                                 show.actuals = TRUE,
                                 ylim = NULL,
                                 gap = 0, 
                                 ...) {
  ## Plot the conditional mean of each series given all observed data.  The
  ## conditional mean includes contributions from shared latent state,
  ## regression effects, and series-specific latent state, if present.
  ## 
  ## Args:
  ##   mbsts.object:  The model object to be plotted.
  ##   series.id: Which series should be plotted?  An integer, logical, or
  ##     character vector.
  ##   same.scale: Should all series be plotted on the same scale?
  ##   burn: The number of MCMC iterations in 'mbsts.object' that should be
  ##     discarded as burn-in.
  ##   time: An optional vector of timestamps to be used as the horizontal axis.
  ##   show.actuals: Logical value indicating whether the actual observed values
  ##     for each time series should be plotted.
  ##   ylim: Limits on the vertical axis.  If ylim is supplied same.scale is
  ##     automatically set to TRUE.
  ##   gap:  The number of lines of space to leave between plots.
  ##   ...:  Extra arguments passed to PlotDynamicDistribution.
  stopifnot(inherits(mbsts.object, "mbsts"))
  original <- LongToWide(mbsts.object$original,
    mbsts.object$series.id,
    mbsts.object$timestamp.info$timestamps)

  if (is.character(series.id)) {
    series.id <- pmatch(series.id, colnames(original))
  }
  if (missing(time)) {
    time <- mbsts.object$timestamp.info$regular.timestamps
  }
  contributions <- mbsts.object$shared.state.contributions
  if (!is.null(series.id)) {
    original <- original[, series.id, drop = FALSE]
    contributions <- contributions[, , series.id, , drop = FALSE]
  } else {
    series.id <- 1:ncol(original)
  }
  state.means <- apply(contributions, c(1, 3, 4), sum)
  labels <- colnames(original)

  predictors <- LongToWideArray(
    mbsts.object$predictors,
    mbsts.object$series.id,
    mbsts.object$timestamp.info$timestamps)
  coefficients <- mbsts.object$regression.coefficients

  niter <- dim(coefficients)[1]
  nseries <- dim(coefficients)[2]
  time.dimension <- dim(predictors)[2]
  regression.effects <- array(0, dim = c(niter, nseries, time.dimension))
  for (m in 1:nseries) {
    regression.effects[, m, ] <- coefficients[, m, ] %*% t(predictors[m, , ])
  }

  regression.effects <- regression.effects[, series.id, , drop = FALSE]

  series.specific.effects <- array(0, dim = c(niter, nseries, time.dimension))
  if (!is.null(mbsts.object$series.specific)) {
    for (m in seq_along(mbsts.object$series.specific)) {
      subordinate.model <- mbsts.object$series.specific[[m]]
      if (!is.null(subordinate.model)) {
        state <- subordinate.model$state.contributions
        state <- rowSums(aperm(state, c(1, 3, 2)), dims = 2)
        series.specific.effects[, m, ] <- series.specific.effects[, m, ] + state
      }
    }
  }
  
  series.specific.effects <-
    series.specific.effects[, series.id, , drop = FALSE]

  state.means <- state.means + regression.effects + series.specific.effects
  
  nplots <- ncol(original)
  plot.rows <- max(1, floor(sqrt(nplots)))
  plot.cols <- ceiling(nplots / plot.rows)

  opar <- par(mfrow = c(plot.rows, plot.cols), mar = rep(gap / 2, 4),
    oma = c(4, 4, 4, 4))
  on.exit(par(opar))
  
  m <- 0
  scale.individually <- !same.scale && is.null(ylim)
  if (!is.null(ylim)) {
    same.scale <- TRUE
  }
  if (is.null(ylim)) {
    ylim <- range(original, state.means, na.rm = TRUE)
  } 

  for (j in 1:plot.rows) {
    for (k in 1:plot.cols) {
      m <- m + 1
      if (m > nplots) {
        
      } else {
        if (scale.individually) {
          ylim <- range(original[, m], state.means[, m, ], na.rm = TRUE)
        }
        PlotDynamicDistribution(curves = state.means[, m, ], timestamps = time,
          axes = FALSE, ylim = ylim, ...)
        box()
        if (!is.null(labels)) {
          text(min(time), max(ylim), labels[m], pos = 4)
        }

        # Add the horizontal axis
        if (IsOdd(k) && j == plot.rows) {
          .AddDateAxis(time, 1)
        } else if (IsEven(k) && j == 1) {
          .AddDateAxis(time, 3)
        }

        # Add the vertical axis
        if (same.scale) {
          if (k == 1 && IsOdd(j)) {
            axis(2)
          } else if (k == plot.cols && IsEven(j)) {
            axis(4)
          }
        }

        if (show.actuals) {
          points(time, original[, m], pch = 20, col = "blue", cex = .2)
        }
      }
    }
  }
}

#===========================================================================
plot.mbsts.prediction <- function(x,
                                  y = NULL,
                                  burn = 0,
                                  plot.original = TRUE,
                                  median.color = "blue",
                                  median.type = 1,
                                  median.width = 3,
                                  interval.quantiles = c(.025, .975),
                                  interval.color = "green",
                                  interval.type = 2,
                                  interval.width = 2,
                                  style = c("dynamic", "boxplot"),
                                  ylim = NULL,
                                  series.id = NULL, 
                                  same.scale = TRUE,
                                  gap = 0, 
                                  ...) {
  ## Plot the results of an mbsts prediction.
  ##
  ## Args:
  ##   x: An object of class mbsts.prediction.
  ##   y: An alias for series.id, see below.
  ##   burn:  The number of MCMC iterations to discard as burn-in.
  ##   plot.original: Logical or numeric.  If TRUE then the prediction is
  ##     plotted after a time series plot of the original series.  If FALSE, the
  ##     prediction fills the entire plot.  If numeric, then it specifies the
  ##     number of trailing observations of the original time series to plot.
  ##   median.color: The color to use for the posterior median of the
  ##     prediction.
  ##   median.type: The type of line (lty) to use for the posterior median of
  ##     the prediction.
  ##   median.width: The width of line (lwd) to use for the posterior median of
  ##     the prediction.
  ##   interval.quantiles: The lower and upper limits of the credible interval
  ##     to be plotted.
  ##   interval.color: The color to use for the upper and lower limits of the
  ##     95% credible interval for the prediction.
  ##   interval.type: The type of line (lty) to use for the upper and lower
  ##     limits of the 95% credible inerval for of the prediction.
  ##   interval.width: The width of line (lwd) to use for the upper and lower
  ##     limits of the 95% credible inerval for of the prediction.
  ##   style: What type of plot should be produced?  A dynamic distribution
  ##     plot, or a time series boxplot.
  ##   ylim:  Limits on the vertical axis.
  ##   series.id: A factor, string, or integer used to indicate which of the
  ##     multivariate series to plot.  If NULL then predictions for all series
  ##     will be plotted.  If there are many series this can make the plot
  ##     unreadable.
  ##   same.scale: Logical.  If TRUE then all predictions are plotted with the
  ##     same scale, and limits are drawn on the Y axis. If FALSE then each
  ##     prediction is drawn to fill its plot region, and no tick marks are
  ##     drawn on the y axis.  If ylim is specified then it is used for all
  ##     plots, and same.scale is ignored.
  ##   gap: The amount of space to leave between plots, measured in lines of
  ##     text.
  ##   ...: Extra arguments to be passed to PlotDynamicDistribution(),
  ##     TimeSeriesBoxplot(), or lines().
  prediction <- x
  nseries <- nrow(prediction$mean)
  if (!is.null(series.id) && is.null(y)) {
    y <- series.id
  }
  series.id <- y
  
  if (is.null(series.id)) {
    series.id <- 1:nseries
  } else {
    if (is.logical(series.id)) {
      stopifnot(length(series.id) == nseries)
    } else if (is.numeric(series.id)) {
      stopifnot(series.id == unique(series.id),
        all(series.id %in% 1:nseries))
    } else if (is.character(series.id)) {
      series.id <- pmatch(series.id, dimnames(prediction$mean)[[1]], nomatch = "")
      if (any(series.id == "")) {
        stop("Some series names did not match.")
      }
    }
    
    prediction$mean <- prediction$mean[series.id, , drop = FALSE]
    prediction$median <- prediction$median[series.id, , drop = FALSE]
    prediction$distribution <- prediction$distribution[, series.id, ,
      drop = FALSE]
    prediction$original.series <- prediction$original.series[, series.id,
      drop = FALSE]
  }

  if (burn > 0) {
    prediction$distribution <-
      prediction$distribution[-(1:burn), , , drop = FALSE]
    prediction$median <- apply(prediction$distribution, c(2, 3), median)
  }
  prediction$interval <- aperm(apply(prediction$distribution, c(2, 3),
    quantile, interval.quantiles), c(2, 1, 3))

  original.series <- prediction$original.series
  if (is.numeric(plot.original)) {
    original.series <- tail(original.series, plot.original)
    plot.original <- TRUE
  }
  n1 <- tail(dim(prediction$distribution), 1)

  # The predict() method stores the prediction as a zoo object.
  time <- index(original.series)
  deltat <- tail(diff(tail(time, 2)), 1)

  nseries.subset <- ncol(original.series)
  plot.rows <- max(1, floor(sqrt(nseries.subset)))
  plot.cols <- ceiling(nseries.subset / plot.rows)

  opar <- par(
    mfrow = c(plot.rows, plot.cols),
    mar = rep(gap / 2, 4),
    oma = c(4, 4, 4, 4))
  on.exit(par(opar))

  series.names <- colnames(prediction$original.series)

  scale.individually <- !same.scale && is.null(ylim)
  if (is.null(ylim)) {
    if (same.scale) {
      original.ylim <- range(original.series, prediction$distribution, na.rm = TRUE)
      ylim <- original.ylim
    } else {
      original.ylim <- NULL
      ylim <- NULL
    }
  } else {
    original.ylim <- ylim
  }

  series <- 0  
  pred.time <- tail(time, 1) + (1:n1) * deltat
  for (row in 1:plot.rows) {
    for (col in 1:plot.cols) {
      series <- series + 1
      if (series <= nseries.subset) {
        if (scale.individually) {
          ylim <- range(prediction$distribution[series, , ],
            original.series[, series],
            na.rm = TRUE)
        } else {
          ylim <- original.ylim
        }

        if (plot.original) {
          plot(time,
            original.series[, series],
            type = "l",
            xlim = range(time, pred.time, na.rm = TRUE),
            ylim = ylim,
            ylab = series.names[series],
            axes = FALSE,
            ...)
          box()
        } else {
          pred.time <- tail(time, 1) + (1:n1) * deltat
        }
        
        style <- match.arg(style)
        if (style == "dynamic") {
          PlotDynamicDistribution(curves = prediction$distribution[, series, ],
            timestamps = pred.time,
            add = plot.original,
            ylim = ylim,
            ylab = series.names[series],
            axes = FALSE,
            ...)
        } else {
          TimeSeriesBoxplot(prediction$distribution[, series, ],
            time = pred.time,
            add = plot.original,
            ylim = ylim,
            ylab = series.names[series],
            axes = FALSE,
            ...)
        }
        lines(pred.time, prediction$median[series, ], col = median.color,
          lty = median.type, lwd = median.width, ...)
        for (i in 1:length(interval.quantiles)) {
          lines(pred.time, prediction$interval[series, i, ], col = interval.color,
            lty = interval.type, lwd = interval.width, ...)
        }
      } else {
        total.time <- c(time, pred.time)
        plot(total.time, rep(0, length(total.time)), type = "n", axes = FALSE)
      }

      if (IsEven(row) && (col == 1)) {
        axis(2)
      } else if(IsOdd(row) && (col == plot.cols)) {
        axis(4)
      }

      if (!scale.individually) {
        if (IsOdd(col) && (row == plot.rows)) {
          .AddDateAxis(time, 1)
        } else if (IsEven(col) && (row == 1)) {
          .AddDateAxis(time, 3)
        }
      }
    }
  }
}
  
