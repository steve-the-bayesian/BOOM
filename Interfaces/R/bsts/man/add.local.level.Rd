% Copyright 2012 Google Inc. All Rights Reserved.
% Author: steve.the.bayesian@gmail.com (Steve Scott)

\name{add.local.level}
\alias{AddLocalLevel}

\Rdversion{1.1}
\title{
  Local level trend state component
}

\description{
  Add a local level model to a state specification.
  The local level model assumes the trend is a
    random walk: \deqn{\alpha_{t+1} = \alpha_t + \epsilon_t \qquad
      \epsilon_t \sim \mathcal{N}(0,\sigma).}{%
      alpha[t+1] = alpha[t] + rnorm(1, 0, sigma). }
    The prior is on the \eqn{\sigma}{sigma}
    parameter.
}

\usage{
  AddLocalLevel(
     state.specification,
     y,
     sigma.prior,
     initial.state.prior,
     sdy,
     initial.y)

}

\arguments{
  \item{state.specification}{A list of state components that you wish to add to.  If
    omitted, an empty list will be assumed.  }

  \item{y}{ The time series to be modeled, as a numeric vector.}

  \item{sigma.prior}{An object created by \code{\link[Boom]{SdPrior}}
    describing the prior distribution for the standard deviation of the
    random walk increments.}

  \item{initial.state.prior}{An object created using
    \code{\link[Boom]{NormalPrior}}, describing the prior distribution
    of the initial state vector (at time 1).}

  \item{sdy}{The standard deviation of the series to be modeled.  This
    will be ignored if \code{y} is provided, or if all the required
    prior distributions are supplied directly.  }

  \item{initial.y}{The initial value of the series being modeled.  This will be
    ignored if \code{y} is provided, or if the priors for the initial
    state are all provided directly.}
}

\value{ Returns a list with the elements necessary to specify a local
  linear trend state model.}

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

\keyword{models}
