% Copyright 2012 Google Inc. All Rights Reserved.
% Author: steve.the.bayesian@gmail.com (Steve Scott)

\name{add.seasonal}
\alias{AddSeasonal}

\Rdversion{1.1}
\title{
  Seasonal State Component
}

\description{
  Add a seasonal model to a state specification.

  The seasonal model can be thought of as a regression on
  \code{nseasons} dummy variables with coefficients constrained to sum
  to 1 (in expectation).  If there are \code{S} seasons then the state
  vector \eqn{\gamma}{gamma} is of dimension \code{S-1}.  The first
  element of the state vector obeys
  \deqn{\gamma_{t+1, 1} = -\sum_{i = 2}^S \gamma_{t, i} + \epsilon_t
  \qquad \epsilon_t \sim \mathcal{N}(0, \sigma)}{%
    gamma[t+1, 1] =  -1 * sum(gamma[t, -1]) + rnorm(1, 0, sigma)
  }
}

\usage{
  AddSeasonal(
     state.specification,
     y,
     nseasons,
     season.duration = 1,
     sigma.prior,
     initial.state.prior,
     sdy)
}

\arguments{
  \item{state.specification}{A list of state components that you wish to add to.  If
    omitted, an empty list will be assumed.  }

  \item{y}{ The time series to be modeled, as a numeric vector.}

  \item{nseasons}{The number of seasons to be modeled.}

  \item{season.duration}{The number of time periods in each season.}

  \item{sigma.prior}{An object created by \code{\link[Boom]{SdPrior}}
    describing the prior distribution for the standard deviation of the
    random walk increments.}

  \item{initial.state.prior}{An object created using
    \code{\link[Boom]{NormalPrior}}, describing the prior distribution
    of the the initial state vector (at time 1).}

  \item{sdy}{The standard deviation of the series to be modeled.  This
    will be ignored if \code{y} is provided, or if all the required
    prior distributions are supplied directly.  }

}

\value{Returns a list with the elements necessary to specify a seasonal
  state model.}

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
  \code{\link{bsts}}.
  \code{\link[Boom]{SdPrior}}
  \code{\link[Boom]{NormalPrior}}
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
