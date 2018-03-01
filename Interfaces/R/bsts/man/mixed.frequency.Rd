% Copyright 2011 Google Inc. All Rights Reserved.
% Author: stevescott@google.com (Steve Scott)

\name{mixed.frequency}

\alias{bsts.mixed}

\title{Models for mixed frequency time series}

\Rdversion{1.0}

\description{
  Fit a structured time series to mixed frequncy data.
}

\usage{
   bsts.mixed(target.series,
              predictors,
              which.coarse.interval,
              membership.fraction,
              contains.end,
              state.specification,
              regression.prior,
              niter,
              ping = niter / 10,
              seed = NULL,
              truth = NULL,
              ...)
}

\arguments{

  \item{target.series}{A vector object of class \code{\link[zoo]{zoo}}
    indexed by calendar dates.  The date associated with each element is
    the LAST DAY in the time interval measured by the corresponding value.
    The value is what Harvey (1989) calls a 'flow' variable.  It is a
    number that can be viewed as an accumulation over the measured time
    interval.  }

  \item{predictors}{ A matrix of class \code{\link[zoo]{zoo}} indexed by
    calendar dates.  The date associated with each row is the LAST DAY
    in the time interval encompasing the measurement.  The dates are
    expected to be at a finer scale than the dates in
    \code{target.series}.  Any predictors should be at sufficient lags
    to be able to predict the rest of the cycle.}

  \item{which.coarse.interval}{A numeric vector of length
    \code{nrow(predictors)} giving the index of the coarse interval
    corresponding to the end of each fine interval.}

  \item{membership.fraction}{A numeric vector of length
    \code{nrow(predictors)} giving the fraction of activity attributed
    to the coarse interval corresponding to the beginning of each fine
    interval.  This is always positive, and will be 1 except when a fine
    interval spans the boundary between two coarse intervals.}

  \item{contains.end}{A logical vector of length \code{nrow(predictors)}
    indicating whether each fine interval contains the end of a coarse
    interval.}

  \item{state.specification}{A state specification like that required
    by \code{bsts}.}

  \item{regression.prior}{A prior distribution created by
    \code{\link[BoomSpikeSlab]{SpikeSlabPrior}}.  A default prior will be generated
    if none is specified.}

  \item{niter}{The desired number of MCMC iterations.}

  \item{ping}{An integer indicating the frequency with which progress
    reports get printed.  E.g. setting \code{ping = 100} will print a
    status message with a time and iteration stamp every 100 iterations.
    If you don't want these messages set \code{ping < 0}.}

  \item{seed}{An integer to use as the random seed for the underlying
    C++ code.  If \code{NULL} then the seed will be set using the
    clock.}

  \item{truth}{For debugging purposes only.  A list containing one or
    more of the following elements.  If any are present then
    corresponding values will be held fixed in the MCMC algorithm.

    \itemize{

      \item A matrix named \code{state} containing the state of the
      coarse model from a fake-data simulation.

      \item A vector named \code{beta} of regression coefficients.

      \item A scalar named \code{sigma.obs}.

    }

  }

  \item{\dots}{Extra arguments passed to SpikeSlabPrior}
}

\value{
  An object of class \code{bsts.mixed}, which is a list with the
  following elements.  Many of these are arrays, in which case the first
  index of the array corresponds to the MCMC iteration number.
  \item{coefficients}{A matrix containing the MCMC draws of the
    regression coefficients.  Rows correspond to MCMC draws, and columns
    correspond to variables.}

  \item{sigma.obs}{The standard deviation of the weekly latent
    observations. }

  \item{state.contributions}{A three-dimensional array containing the
    MCMC draws of each state model's contributions to the state of the
    weekly model.  The three dimensions are MCMC iteration, state model,
    and week number.}

  \item{weekly}{A matrix of MCMC draws of the weekly latent
    observations.  Rows are MCMC iterations, and columns are weekly time
    points.}

  \item{cumulator}{A matrix of MCMC draws of the cumulator variable.}

  The returned object also contains MCMC draws for the parameters of the
  state models supplied as part of \code{state.specification}, relevant
  information passed to the function call, and other supplemental
  information.

}
\references{
  Harvey (1990), "Forecasting, structural time series, and the Kalman
  filter", Cambridge University Press.

  Durbin and Koopman (2001), "Time series analysis by state space
  methods", Oxford University Press.
}

\author{
  Steven L. Scott \email{stevescott@google.com}
}

\seealso{
  \code{\link{bsts}},
  \code{\link{AddLocalLevel}},
  \code{\link{AddLocalLinearTrend}},
  \code{\link{AddGeneralizedLocalLinearTrend}},
  \code{\link[BoomSpikeSlab]{SpikeSlabPrior}},
  \code{\link[Boom]{SdPrior}}.
}

\examples{
  data <- SimulateFakeMixedFrequencyData(nweeks = 104, xdim = 20)

  ## Setting an upper limit on the standard deviations can help keep the
  ## MCMC from flying off to infinity.
  sd.limit <- sd(data$coarse.target)
  state.specification <-
       AddLocalLinearTrend(list(),
                     data$coarse.target,
                     level.sigma.prior = SdPrior(1.0, 5, upper.limit = sd.limit),
                     slope.sigma.prior = SdPrior(.5, 5, upper.limit = sd.limit))
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
                      niter = 500,
                      expected.r2 = .999,
                      prior.df = 1)

  plot(model, "state")
  plot(model, "components")

}
\keyword{models}
\keyword{regression}
