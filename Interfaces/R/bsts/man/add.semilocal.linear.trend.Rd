% Copyright 2012 Google Inc. All Rights Reserved.
% Author: stevescott@google.com (Steve Scott)

\name{add.semilocal.linear.trend}
\alias{AddGeneralizedLocalLinearTrend}
\alias{AddSemilocalLinearTrend}

\Rdversion{1.1}
\title{Semilocal Linear Trend}

\description{

  The semi-local linear trend model is similar to the local linear
  trend, but more useful for long-term forecasting.  It assumes the
  level component moves according to a random walk, but the slope
  component moves according to an AR1 process centered on a
  potentially nonzero value \eqn{D}{D}.  The equation for the level is

  \deqn{\mu_{t+1} = \mu_t + \delta_t + \epsilon_t \qquad \epsilon_t \sim
    \mathcal{N(0, \sigma_\mu)}.}{ %
    mu[t+1] = mu[t] + delta[t] + rnorm(1, 0, sigma.level)
  }

  The equation for the slope is \deqn{\delta_{t+1} = D + \phi
    (\delta_t - D) + \eta_t \qquad \eta_t \sim \mathcal{N(0,
      \sigma_\delta)}.}{%
    delta[t+1] = D + phi * (delta[t] - D) + rnorm(1, 0, sigma.slope).}

  This model differs from the local linear trend model in that the
  latter assumes the slope \eqn{\delta_t}{delta[t]} follows a random
  walk.  A stationary AR(1) process is less variable than a random walk
  when making projections far into the future, so this model often gives
  more reasonable uncertainty estimates when making long term forecasts.

  The prior distribution for the semi-local linear trend has four
  independent components.  These are:

  \itemize{

    \item{an inverse gamma prior on the level
      standard deviation \eqn{\sigma_\mu}{sigma.level},}

    \item{an inverse gamma prior on the slope standard deviation
      \eqn{\sigma_\delta}{sigma.slope}, }

    \item{a Gaussian prior on the long run slope parameter \eqn{D}{D},}

    \item{and a potentially truncated Gaussian prior on the AR1
      coefficient \eqn{\phi}{phi}.  If the prior on \eqn{\phi}{phi} is
      truncated to (-1, 1), then the slope will exhibit short term
      stationary variation around the long run slope \eqn{D}{D}.}
  }

}

\usage{
  AddSemilocalLinearTrend(
     state.specification = list(),
     y = NULL,
     level.sigma.prior = NULL,
     slope.mean.prior = NULL,
     slope.ar1.prior = NULL,
     slope.sigma.prior = NULL,
     initial.level.prior = NULL,
     initial.slope.prior = NULL,
     sdy = NULL,
     initial.y = NULL)
}

\arguments{
  \item{state.specification}{A list of state components that you wish to add to.  If
    omitted, an empty list will be assumed.  }

  \item{y}{The time series to be modeled, as a numeric vector.  This can
    be omitted if \code{sdy} and \code{initial.y} are supplied, or if
    all prior distributions are supplied directly.}

 \item{level.sigma.prior}{An object created by
   \code{\link[Boom]{SdPrior}} describing the prior
   distribution for the standard deviation of the level component.}

  \item{slope.mean.prior}{An object created by
   \code{\link[Boom]{NormalPrior}} giving the prior distribution for
   the mean parameter in the generalized local linear trend model (see
   below). }

  \item{slope.ar1.prior}{An object created by
   \code{\link[Boom]{Ar1CoefficientPrior}} giving the prior
   distribution for the ar1 coefficient parameter in the generalized
   local linear trend model (see below). }

 \item{slope.sigma.prior}{An object created by
   \code{\link[Boom]{SdPrior}} describing the prior distribution of
   the standard deviation of the slope component.}

  \item{initial.level.prior}{An object created by
    \code{\link[Boom]{NormalPrior}} describing the initial distribution
    of the level portion of the initial state vector.}

  \item{initial.slope.prior}{An object created by
    \code{\link[Boom]{NormalPrior}} describing the prior distribution
    for the slope portion of the initial state vector.}

  \item{sdy}{The standard deviation of the series to be modeled.  This
    will be ignored if \code{y} is provided, or if all the required
    prior distributions are supplied directly.  }

  \item{initial.y}{The initial value of the series being modeled.  This will be
    ignored if \code{y} is provided, or if the priors for the initial
    state are all provided directly.}
}

\value{ Returns a list with the elements necessary to specify a
  generalized local linear trend state model.}

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
}

\keyword{models}
