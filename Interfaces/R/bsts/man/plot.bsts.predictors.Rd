% Copyright 2011 Google Inc. All Rights Reserved.
% Author: steve.the.bayesian@gmail.com (Steve Scott)

\name{plot.bsts.predictors}
\title{Plot the most likely predictors}

\alias{PlotBstsPredictors}

\description{Creates a time series plot showing the most likely
  predictors of a time series used to fit a \code{\link{bsts}} object.}

\usage{
  PlotBstsPredictors(bsts.object,
                     burn = SuggestBurn(.1, bsts.object),
                     inclusion.threshold = .1,
                     ylim = NULL,
                     flip.signs = TRUE,
                     show.legend = TRUE,
                     grayscale = TRUE,
                     short.names = TRUE,
                     ...)
}

\arguments{
  \item{bsts.object}{An object of class \code{\link{bsts}}.}

  \item{burn}{The number of observations you wish to discard as burn-in.}

  \item{inclusion.threshold}{Plot predictors with marginal inclusion
    probabilities above this threshold.}

  \item{ylim}{Scale for the vertical axis.  }

  \item{flip.signs}{If true then a predictor with a negative sign will
    be flipped before being plotted, to better align visually
    with the target series.}

  \item{show.legend}{
    Should a legend be shown indicating which predictors are plotted?
  }

  \item{grayscale}{Logical.  If \code{TRUE} then lines for different
    predictors grow progressively lighter as their inclusion probability
    decreases.  If \code{FALSE} then lines are drawn in black.}

  \item{short.names}{Logical.  If \code{TRUE} then a common prefix or
    suffix shared by all the variables will be discarded.}

  \item{...}{Extra arguments to be passed to \code{\link{plot}}.}
}

\examples{
  data(AirPassengers)
  y <- log(AirPassengers)
  lag.y <- c(NA, head(y, -1))
  ss <- AddLocalLinearTrend(list(), y)
  ss <- AddSeasonal(ss, y, nseasons = 12)
  ## Call bsts with na.action = na.omit to omit the leading NA in lag.y
  model <- bsts(y ~ lag.y, state.specification = ss, niter = 500,
                na.action = na.omit)
  plot(model, "predictors")
}

\seealso{
  \code{\link{bsts}}
  \code{\link[Boom]{PlotDynamicDistribution}}
  \code{\link[BoomSpikeSlab]{plot.lm.spike}}
}
