library(Boom)

### Running student_mvss_regression_test locally from the main BOOM directory
### will generate a bunch of files containing the MCMC output from the tests.
### Loading this file into an R session with the main BOOM directory as its
### working directory will allow you to visualize the results of the MCMC.

PlotParamDensity <- function(fname, same.scale=TRUE, burn = 0) {
  ## A density of a set of parameters in a file.
  ## Args:
  ##   fname: The name of the file. The columns are the different parameters.
  ##     Rows are different draws.  The first line is the file contains the
  ##     "true" values.
  ##   same.scale: If TRUE then all the density plots are on the same scale.
  ##     If FALSE then each plot is scaled separately.
  ##   burn: If positive, the number of iterations to discard as
  ##     burn-in. Ignored otherwise.
  draws <- mscan(fname)
  truth <- draws[1, ]
  mcmc <- draws[-1, ]
  if (burn > 0) {
    mcmc <- mcmc[-(1:burn), ]
  }
  nparams <- length(truth)
  opar <- par(mfrow=c(1, nparams))
  on.exit(par(opar))
  for (i in 1:nparams) {
    plot(density(mcmc[, i]), xlim=range(draws))
    abline(v=truth[i], lwd=3)
    abline(h=0, lty=3)
  }
}

PlotParamTs <- function(fname, same.scale=TRUE) {
  ## A time series plot of a set of parameters in a file.
  ## Args:
  ##   fname: The name of the file. The columns are the different parameters.
  ##     Rows are different draws.  The first line is the file contains the
  ##     "true" values.
  ##   same.scale: If TRUE then all the time series plots are on the same scale.
  ##     If FALSE then each plot is scaled separately.
  draws <- mscan(fname)
  truth <- draws[1, ]
  mcmc <- draws[-1, ]
  nparams <- length(truth)
  PlotManyTs(mcmc, truth=truth, same.scale=same.scale)
}
## PlotParamTs("TrendSeasonalMcmcTest.residual_sd.draws")
## PlotParamTs("TrendSeasonalMcmcTest.tail_thickness.draws", same.scale=T)
## PlotParamDensity("TrendSeasonalMcmcTest.tail_thickness.draws")

## PlotParamTs("McmcTest.residual_sd.draws")
## PlotParamDensity("McmcTest.residual_sd.draws")

PlotObservationCoefficients <- function(series=0:9, seasonal = TRUE, refline = 0, ...) {
  if (seasonal) {
    base.name <- "TrendSeasonalMcmcTest.seasonal_observation_coefficients_series_"
  } else {
    base.name <- "TrendSeasonalMcmcTest.trend_observation_coefficients_series_"
  }

  truth <- numeric(length(series))
  draws <- list()
  for (i in 1:length(series)) {
    fname <- paste0(base.name, series[i])
    coef <- mscan(fname)
    draws[[paste(i)]] <- coef[-1, 1]
    truth[i] <- coef[1, 1]
  }
  draws <- as.data.frame(draws)
  PlotManyTs(draws, truth=truth, gap=1, refline = refline, ...)
  return(invisible(rbind(truth, as.matrix(draws))))
}

PlotSeasonal <- function(iter = NULL, times = NULL, ylim = NULL, ...) {
  fname <- paste0("TrendSeasonalMcmcTest.seasonal_state.draws")
  draws <- mscan(fname)
  if (is.null(ylim)) {
    ylim <- range(draws)
  }
  truth <- draws[1, ]
  draws <- draws[-1, ]

  if (!is.null(iter)) {
    draws <- draws[iter, ]
  }

  if (!is.null(times)) {
    draws <- draws[, times]
    truth <- truth[times]
  }

  opar <- par(mfrow = c(1,2))
  on.exit(par(opar))

  PlotDynamicDistribution(draws, ylim = ylim, ...)
  lines(truth, col = "green", lwd = 2)

  centered.draws <- t(t(draws) - truth)
  PlotDynamicDistribution(centered.draws)
}

PlotStateContribution <- function(series, iter = NULL, times = NULL) {
  trend.fname <- paste0("TrendSeasonalMcmcTest.trend_contribution_series_", series)
  trend.draws <- mscan(trend.fname)
  true.trend <- trend.draws[1, ]
  trend <- trend.draws[-1, ]
  if (!is.null(iter)) {
    trend <- trend[iter, ]
  }
  if (!is.null(times)) {
    trend <- trend[, times]
    true.trend <- true.trend[times]
  }


  seasonal.fname <- paste0("TrendSeasonalMcmcTest.seasonal_contribution_series_", series)
  seasonal.draws <- mscan(seasonal.fname)
  true.seasonal <- seasonal.draws[1, ]
  seasonal <- seasonal.draws[-1, ]
  if (!is.null(iter)) {
    seasonal <- seasonal[iter, ]
  }
  if (!is.null(times)) {
    seasonal <- seasonal[, times]
    true.seasonal <- true.seasonal[times]
  }

  ## reg.fname <- paste0("TrendSeasonalMcmcTest.regression_contribution_series_", series)
  ## reg.draws <- mscan(reg.fname)
  ## true.reg <- reg.draws[1, ]
  ## reg <- reg.draws[-1, ]

  true.state <- true.trend + true.seasonal # + true.reg
  state <- trend + seasonal # + reg

  opar <- par(mfrow=c(3,2))
  on.exit(par(opar))

  PlotDynamicDistribution(trend, ylim = range(trend, true.trend), main="Trend Distribution")
  lines(true.trend, col = "green", lwd=2)

  centered.trend <- t(t(trend) - true.trend)
  PlotDynamicDistribution(centered.trend, main = "Trend Residual Distribution")

  PlotDynamicDistribution(seasonal, ylim = range(seasonal, true.seasonal),
                          main="Seasonal Distribution")
  lines(true.seasonal, col = "green", lwd=2)

  centered.seasonal <- t(t(seasonal) - true.seasonal)
  PlotDynamicDistribution(centered.seasonal, main = "Seasonal Residual Distribution")

#   PlotDynamicDistribution(reg)
#   lines(true.reg, col = "green")

  PlotDynamicDistribution(state, ylim = range(state, true.state),
                          main = "Overall State Distribution")
  lines(true.state, col = "green", lwd=2)

  centered.state <- t(t(state) - true.state)
  PlotDynamicDistribution(centered.state, main = "State Residual Distribution")
}

PlotXtwy <- function(fname = "foo", burn = 0, ...) {
  ## After a call to
  ##
  ## grep xtwy_ foo | awk -F'=' '{print $2}' > xtwy
  ##
  ## this call plots the time series of results.
  system(paste0(
    "grep xtwy_ ", fname, " | awk -F'=' '{print $2}' > xtwy"))
  xtwy <- matrix(scan("xtwy"), ncol=3, byrow=T)
  if (burn > 0) {
    xtwy <- xtwy[-(1:burn), , drop = FALSE]
  }
  PlotManyTs(xtwy, ...)
}

PlotXtwx <- function(fname = "foo", burn = 0, ...) {
  cmd <- paste0("grep -A 1 xtwx_ ", fname, " | grep -v 'xtwx_' > xtwx")
  cat(cmd)
  system(cmd)
  raw <- scan("xtwx", na.strings = "--")
  raw <- raw[!is.na(raw)]
  xtwx <- matrix(raw, ncol=3, byrow=T)
  if (burn > 0) {
    xtwx <- xtwx[-(1:burn), , drop = FALSE]
  }
                                        # PlotManyTs(xtwx, ...)
  pairs(xtwx)
}
