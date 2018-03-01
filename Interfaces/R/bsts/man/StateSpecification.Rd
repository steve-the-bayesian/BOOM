% Copyright 2011 Google Inc. All Rights Reserved.
% Author: stevescott@google.com (Steve Scott)

\name{StateSpecification}

\alias{state.specification}
\alias{StateSpecification}

\Rdversion{1.1}

\title{
  Add a state component to a Bayesian structural time series model
}

\description{
  Add a state component to the \code{state.specification} argument in a
  \code{\link{bsts}} model.
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
  \code{\link{bsts}}.
  \code{\link[Boom]{SdPrior}}
  \code{\link[Boom]{NormalPrior}}
  \code{\link[Boom]{Ar1CoefficientPrior}}
}

\examples{
  data(AirPassengers)
  y <- log(AirPassengers)
  ss <- AddLocalLinearTrend(list(), y)
  ss <- AddSeasonal(ss, y, nseasons = 12)
  model <- bsts(y, state.specification = ss, niter = 500)
  pred <- predict(model, horizon = 12, burn = 100)
  plot(pred)
}
\keyword{models}
\keyword{regression}
