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

  ### TODO: allow for subsetting in regression effects.

  ### TODO: add in series-specific effects.

  ### DONTSUBMIT
  
  state.means <- state.means + regression.effects
  
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

  .AddTimeAxis <- function(side, time) {
    if (inherits(time, "Date")) {
      axis.Date(side, time, xpd = NA)
    } else if (inherits(time, "POSIXt")) {
      axis.POSIXct(1, as.POSIXct(time), xpd = NA)
    } else {
      axis(side, xpd = NA)
    }
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
          .AddTimeAxis(1, time)
        } else if (IsEven(k) && j == 1) {
          .AddTimeAxis(3, time)
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
