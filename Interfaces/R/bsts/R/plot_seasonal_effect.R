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

PlotSeasonalEffect <- function(bsts.object,
                               nseasons = 7,
                               season.duration = 1,
                               same.scale = TRUE,
                               ylim = NULL,
                               get.season.name = NULL,
                               burn = SuggestBurn(.1, bsts.object),
                               ...) {
  ## Creates a set of plots similar to a 'month plot' showing how the
  ## effect of each season has changed over time.  This function uses
  ## mfrow to create a page of plots, so it cannot be used on the same
  ## page as other plotting functions.
  ##
  ## Args:
  ##   bsts.object:  A bsts model containing a seasonal component.
  ##   ylim:  The limits of the vertical axis.
  ##   same.scale: Used only if ylim is NULL.  If TRUE then all
  ##     figures are plotted on the same scale.  If FALSE then each
  ##     figure is independently scaled.
  ##   nseasons:  The number of seasons in the seasonal component to be plotted.
  ##   season.duration: The duration of each season in the seasonal
  ##     component to be plotted.
  ##   get.season.name: A function taking a Date, POSIXt, or other
  ##     time object used as the index of the original data series
  ##     used to fit 'bsts.object,' and returning a character string
  ##     that can be used as a title for each panel of the plot.  If
  ##     this is NULL and nseasons is one of the following time units
  ##     then the associated function will be used. (see ?weekdays)
  ##     - 4  quarters
  ##     - 7  weekdays
  ##     - 12 months
  ##   burn: The number of MCMC iterations to be discarded as burn-in.
  ##   ...:  Extra arguments passed to PlotDynamicDistribution.
  ##
  ## Returns:
  ##   Invisible NULL.
  ##
  effect.names <- dimnames(bsts.object$state.contributions)$component
  position <- grep("seasonal", effect.names)
  if (length(position) == 1) {
    name.components <- strsplit(effect.names[position], ".", fixed = TRUE)[[1]]
    nseasons <- as.numeric(name.components[2])
    season.duration <- as.numeric(name.components[3])
  } else {
    effect.name <- paste("seasonal", nseasons, season.duration, sep = ".")
    position <- grep(effect.name, effect.names)
  }
  if (length(position) != 1) {
    stop("The desired seasonal effect could not be located. ",
         "Did you specify 'nseasons' and 'season.duration' correctly?")
  }
  effect <- bsts.object$state.contributions[, position, ]
  if (burn > 0) {
    effect <- effect[-(1:burn), , drop = FALSE]
  }
  if (is.null(ylim) && same.scale == TRUE) {
    ylim <- range(effect)
  }
  vary.ylim <- is.null(ylim)
  time <- bsts.object$timestamp.info$regular.timestamps
  nr <- floor(sqrt(nseasons))
  nc <- ceiling(nseasons / nr)
  if (is.null(get.season.name) && inherits(time, c("Date", "POSIXt"))) {
    if (nseasons == 7) {
      get.season.name <- weekdays
    } else if (nseasons == 12) {
      get.season.name <- months
    } else if (nseasons == 4) {
      get.season.name <- quarters
    }
  }

  opar <- par(mfrow = c(nr, nc))
  on.exit(par(opar))
  for (season in 1:nseasons) {
    time.index <- seq(from = 1 + (season - 1) * season.duration,
                      to = length(time),
                      by = nseasons * season.duration)
    season.effect <- effect[, time.index]
    if (vary.ylim) {
      ylim <- range(season.effect)
    }
    dates <- time[time.index]
    PlotDynamicDistribution(season.effect,
                            dates,
                            ylim = ylim,
                            xlim = range(time),
                            ...)
    lines(dates, apply(season.effect, 2, median), col = "green")
    if (inherits(dates, c("Date", "POSIXt")) && !is.null(get.season.name)) {
      season.name <- get.season.name(dates[1])
      title(main = season.name)
    } else {
      title(main = paste("Season", season))
    }
  }
  return(invisible(NULL))
}

.FindDayOfWeekSpec <- function(bsts.object) {
  ## Find and return the state specification object corresponding to the
  ## day-of-week cycle.  If there are multiple cycles of size 7 the first one is
  ## returned.  If no such state specification exists then NULL is returned.
  ss <- bsts.object$state.specification
  for (i in 1:length(ss)) {
    if (inherits(ss[[i]], "Seasonal")
      && ss[[i]]$nseasons == 7) {
      return(ss[[i]])
    }
  }
  return(NULL)
}

PlotDayOfWeekCycle <- function(bsts.object,
                               burn = SuggestBurn(.1, bsts.object),
                               time = NULL, 
                               ylim = NULL,
                               state.specification = NULL,
                               ...) {
  ## Plots the time series of Sunday, Monday, Tuesday, ... effects in a time
  ## series model that was fit to daily data with a day-of-week seasonal
  ## component.  
  ##
  ## Args:
  ##   bsts.object: A bsts model that was fit using a Seasonal state component
  ##     with a 7-season cycle.
  ##   burn:  The number of MCMC iterations to discard.
  ##   time:  An optional vector of times to plot on the horizontal axis.
  ##   ylim:  Limits for the vertical axis.
  ##   state.specification: A Seasonal state specification object.  If NULL then
  ##     it will be taken from bsts.object$state.specification, and if no
  ##     apprpriate object can befound an error will be raised.
  ##   ...: Extra arguments passed to PlotDynamicDistribution.
  ##
  ## NOTE: split.screen is used to make this plot, so it is incompatible with
  ## other plot multi-plot frameworks, but it can be used with other plots in a
  ## split.screen environment.
  ##
  ## Returns:
  ##   Invisible(NULL)
  if (is.null(state.specification)) {
    state.specification <- .FindDayOfWeekSpec(bsts.object)
  }
  stopifnot(inherits(state.specification, "Seasonal"),
    state.specification$nseasons == 7)

  position <- grep(state.specification$name,
    dimnames(bsts.object$state.contributions)[[2]])
  if (length(position) != 1) {
    stop("The day of week cycle could not be located, or is not unique.")
  }
  effect <- bsts.object$state.contributions[, position, ]
  if (burn > 0) {
    effect <- effect[-(1:burn), , drop = FALSE]
  }

  nseasons <- 7
  season.duration <- state.specification$season.duration
  if (is.null(time)) {
    time <- bsts.object$timestamp.info$regular.timestamps
  }
  
  screen.matrix <- .ScreenMatrix(nrow = 3, ncol = 3, top.margin = .12)
  ## Each row of screen.numbers corresponds to the coordinates given
  ## screen.matrix.  However, if split.screen was called before this those
  ## numbers might not be 1:9.
  screen.numbers <- split.screen(screen.matrix)
  on.exit(close.screen(screen.numbers))

  ## Define the screens where the different days of the week should appear.
  ## Skip screens 3 and 6.
  season.screen.numbers <- c(
    screen.numbers[1],
    screen.numbers[2],
    screen.numbers[4],
    screen.numbers[5],
    screen.numbers[7],
    screen.numbers[8],
    screen.numbers[9])

  if (is.null(ylim)) {
    ylim <- range(effect)
  }

  for (season in 1:nseasons) {
    screen(season.screen.numbers[season])
    par(mar = rep(0, 4))
    time.index <- seq(from = 1 + (season - 1) * season.duration,
                      to = length(time),
                      by = nseasons * season.duration)

    season.effect <- effect[, time.index]
    dates <- time[time.index]
    PlotDynamicDistribution(season.effect,
                            dates,
                            ylim = ylim,
                            xlim = range(time),
                            axes = FALSE,
                            ylab = "",
                            xlab = "",
                            ...)
    abline(h = 0, lty = 3)
    lines(dates, apply(season.effect, 2, median), col = "green")
    if (inherits(dates, c("Date", "POSIXt"))) {
      title(main = weekdays(dates[1]), line = -2, cex.main = .6)
    } else {
      title(main = paste("Season", season), line = -2, cex.main = .6)
    }
    if (season %in% c(5, 7)) {
      ## On the bottom row, so put in the time axis.
      .AddDateAxis(time, 1)
    }
    if (season == 2) {
      .AddDateAxis(time, 3)
    }
    if (season == 2 || season == 7) {
      ## Put axis on the right.
      axis(4)
    }
    if (season == 3) {
      ## Put axis on the left.
      axis(2)
    }
  }
  return(invisible(NULL))
}

PlotMonthlyAnnualCycle <- function(bsts.object,
                                   ylim = NULL,
                                   same.scale = TRUE,
                                   burn = SuggestBurn(.1, bsts.object),
                                   ...) {
  ## For models with a MonthlyAnnualCycle state component, plot the time series
  ## of January's, February's, etc.
  ##
  ## Args:
  ##   bsts.object:  The bsts model object containing a MonthlyAnnualCycle.
  ##   ylim:  The limits of the vertical axis.
  ##   same.scale: Used only if ylim is NULL.  If TRUE then all figures are
  ##     plotted on the same scale.  If FALSE then each figure is independently
  ##     scaled.
  ##   burn: The number of MCMC iterations to be discarded as burn-in.
  ##   ...:  Extra arguments passed to PlotDynamicDistribution.
  ##
  ## Returns:
  ##   Draws 12 dynamic distribution plots on the current graphics device (one
  ##   for each month), then returns invisible NULL.
  effect <- bsts.object$state.contributions[, "Monthly", ]
  if (burn > 0) {
    effect <- effect[-(1:burn), , drop = FALSE]
  }
  if (is.null(ylim) && same.scale == TRUE) {
    ylim <- range(effect)
  }
  vary.ylim <- is.null(ylim)
  time <- bsts.object$timestamp.info$regular.timestamps
  opar <- par(mfrow = c(4, 3))
  on.exit(par(opar))
  time.dimension <- ncol(effect)
  dt <- effect[, 2:time.dimension] - effect[, 1:(time.dimension - 1)]
  new.month <- c(TRUE, dt[1, ] != 0)
  time <- time[new.month]
  effect <- effect[, new.month]
  for (m in 1:12) {
    this.month <- months(time) == month.name[m]
    month.effect <- effect[, this.month]
    if (vary.ylim) {
      ylim <- range(month.effect)
    }
    dates <- time[this.month]
    PlotDynamicDistribution(month.effect,
                            dates,
                            ylim = ylim,
                            xlim = range(time),
                            ...)
    lines(dates, apply(month.effect, 2, median), col = "green")
    title(main = month.name[m])
  }
  return(invisible(NULL))
}
