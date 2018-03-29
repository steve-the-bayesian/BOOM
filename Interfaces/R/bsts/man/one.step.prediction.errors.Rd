% Copyright 2011 Google Inc. All Rights Reserved.
% Author: steve.the.bayesian@gmail.com (Steve Scott)

\name{one.step.prediction.errors}
\alias{bsts.prediction.errors}
\title{Prediction Errors}
\Rdversion{1.1}
\description{
  Computes the one-step-ahead prediction errors for a \code{\link{bsts}}
  model.
}

\usage{
bsts.prediction.errors(bsts.object,
                       cutpoints = NULL,
                       burn = SuggestBurn(.1, bsts.object))
}

\arguments{

  \item{bsts.object}{ An object of class \code{\link{bsts}}.}

  \item{cutpoints}{An increasing sequence of integers between 1 and the
    number of time points in the trainig data for \code{bsts.object}, or
    \code{NULL}.  If \code{NULL} then the in-sample one-step prediction
    errors from the bsts object will be extracted and returned.
    Otherwise the model will be re-fit with a separate MCMC run
    for each entry in 'cutpoints'.  Data up to each cutpoint will be
    included in the fit, and one-step prediction errors for data after
    the cutpoint will be computed.  }

  \item{burn}{An integer giving the number of MCMC iterations to discard
    as burn-in.  If \code{burn <= 0} then no burn-in sample will be
    discarded.}
}

\value{
  A matrix of draws of the one-step-ahead prediction errors.  Rows of
  the matrix correspond to MCMC draws.  Columns correspond to time.
}

\details{

 Returns the posterior distribution of the one-step-ahead prediction errors
 from the bsts.object.  The errors are computing using the Kalman filter,
 and are of two types.

 Purely in-sample errors are computed as a by-product of the Kalman
 filter as a result of fitting the model.  These are stored in the
 bsts.object assuming the \code{save.prediction.errors} option is TRUE,
 which is the default (See \code{\link{BstsOptions}}).  The in-sample
 errors are 'in-sample' in the sense that the parameter values used to
 run the Kalman filter are drawn from their posterior distribution given
 complete data.  Conditional on the parameters in that MCMC iteration,
 each 'error' is the difference between the observed y[t] and its
 expectation given data to t-1.

 Purely out-of-sample errors can be computed by specifying the 'cutpoints'
 argument.  If cutpoints are supplied then a separate MCMC is run using just
 data up to the cutpoint.  The Kalman filter is then run on the remaining
 data, again finding the difference between y[t] and its expectation given
 data to t-1, but conditional on parameters estimated using data up to the
 cutpoint.
}

\references{
  Harvey (1990), "Forecasting, structural time series, and the Kalman
  filter", Cambridge University Press.

  Durbin and Koopman (2001), "Time series analysis by state space
  methods", Oxford University Press.
}

\author{
  Steven L. Scott \email{steve.the.bayesian@gmail.com}
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
  data(AirPassengers)
  y <- log(AirPassengers)
  ss <- AddLocalLinearTrend(list(), y)
  ss <- AddSeasonal(ss, y, nseasons = 12)

\dontrun{
  model <- bsts(y, state.specification = ss, niter = 500)
}
\dontshow{
  model <- bsts(y, state.specification = ss, niter = 200)
}
  errors <- bsts.prediction.errors(model, burn = 100)
  PlotDynamicDistribution(errors$in.sample)

  ## Compute out of sample prediction errors beyond times 80 and 120.
  errors <- bsts.prediction.errors(model, cutpoints = c(80, 120))
  plot(model, "prediction.errors", cutpoints = c(80, 120))
  str(errors)     ## three matrices, with 400 ( = 500 - 100) rows
                  ## and length(y) columns
}
\keyword{models}
\keyword{regression}
