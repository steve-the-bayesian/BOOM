% Copyright 2011 Google Inc. All Rights Reserved.
% Author: steve.the.bayesian@gmail.com (Steve Scott)

\name{plot.bsts.mixed}

\title{Plotting functions for mixed frequency Bayesian structural time series}

\alias{plot.bsts.mixed}
\alias{PlotBstsMixedComponents}
\alias{PlotBstsMixedState}

\description{Functions for plotting the output of a mixed frequency time series regression.}

\usage{
  \method{plot}{bsts.mixed}(x,
                            y = c("state", "components",
                                  "coefficients", "predictors", "size"),
                            ...)

   PlotBstsMixedState(bsts.mixed.object,
                      burn = SuggestBurn(.1, bsts.mixed.object),
                      time = NULL,
                      fine.scale = FALSE,
                      style = c("dynamic", "boxplot"),
                      trim.left = NULL,
                      trim.right = NULL,
                      ...)

   PlotBstsMixedComponents(bsts.mixed.object,
                           burn = SuggestBurn(.1, bsts.mixed.object),
                           time = NULL,
                           same.scale = TRUE,
                           fine.scale = FALSE,
                           style = c("dynamic", "boxplot"),
                           layout = c("square", "horizontal", "vertical"),
                           ylim = NULL,
                           trim.left = NULL,
                           trim.right = NULL,
                           ...)
}

\arguments{

  \item{x}{An object of class \code{\link{bsts.mixed}}.}

  \item{bsts.mixed.object}{An object of class \code{\link{bsts.mixed}}.}

  \item{y}{A character string indicating the aspect of the model that
    should be plotted. }

  \item{burn}{The number of MCMC iterations to discard as burn-in.}

  \item{time}{An optional vector of values to plot against.  If missing,
    the default is to obtain the time scale from the original time
    series.}

  \item{fine.scale}{Logical.  If \code{TRUE} then the plots will be at
    the weekly level of granularity.  If \code{FALSE} they will be at
    the monthly level.}

  \item{same.scale}{Logical.  If \code{TRUE} then all the state
    components will be plotted with the same scale on the vertical axis.
    If \code{FALSE} then each component will get its own scale for the
    vertical axis.}

  \item{style}{character.  If "dynamic" then a dynamic distribution plot
    will be shown.  If "box" then boxplots will be shown.}

  \item{layout}{A character string indicating whether the plots showing
    components of state should be laid out in a square, horizontally, or
    vertically.}

  \item{trim.left}{ A logical indicating whether the first (presumedly
    partial) observation in the aggregated state time series should be
    removed.  }

  \item{trim.right}{ A logical indicating whether the last (presumedly
    partial) observation in the aggregated state time series should be
    removed.  }

  \item{ylim}{Limits for the vertical axis.  Optional.}

  \item{...}{Additional arguments to be passed to
    \code{\link[Boom]{PlotDynamicDistribution}} or \code{\link{TimeSeriesBoxplot}}}

}

\details{
  \code{\link{PlotBstsMixedState}} plots the aggregate state
  contribution (including regression effects) to the mean, while
  \code{\link{PlotBstsComponents}} plots the contribution of each state
  component separately.  \code{\link{PlotBstsCoefficients}} creates a
  significance plot for the predictors used in the state space
  regression model.
}

\value{
  These functions are called for their side effect, which is to produce
  a plot on the current graphics device.
}

\examples{
% TODO(stevescott):  fix flaky mixed.frequency examples in .Rd files
\dontrun{
## This example is flaky and needs to be fixed
  data <- SimulateFakeMixedFrequencyData(nweeks = 104, xdim = 20)
  state.specification <- AddLocalLinearTrend(list(), data$coarse.target)
  weeks <- index(data$predictor)
  months <- index(data$coarse.target)
  which.month <- MatchWeekToMonth(weeks, months[1])
  membership.fraction <- GetFractionOfDaysInInitialMonth(weeks)
  contains.end <- WeekEndsMonth(weeks)

  model <- bsts.mixed(target.series = data$coarse.target,
                      predictors = data$predictors,
                      membership.fraction = membership.fraction,
                      contains.end = contains.end,
                      which.coarse = which.month,
                      state.specification = state.specification,
                      niter = 500)

  plot(model, "state")
  plot(model, "components")
}
}

\seealso{
  \code{\link{bsts.mixed}}
  \code{\link[Boom]{PlotDynamicDistribution}}
  \code{\link[BoomSpikeSlab]{plot.lm.spike}}
  \code{\link{PlotBstsSize}}
}
