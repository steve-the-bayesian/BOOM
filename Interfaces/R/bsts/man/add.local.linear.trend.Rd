% Copyright 2012 Google Inc. All Rights Reserved.
% Author: stevescott@google.com (Steve Scott)

\name{add.local.linear.trend}
\alias{AddLocalLinearTrend}

\Rdversion{1.1}
\title{
  Local linear trend state component
}

\description{
  Add a local linear trend model to a state specification.
  The local linear trend model assumes that both the mean and the slope of the
  trend follow random walks.  The equation for the mean is
  \deqn{\mu_{t+1} = \mu_t + \delta_t + \epsilon_t  \qquad \epsilon_t
    \sim \mathcal{N}(0, \sigma_\mu).}{%
    mu[t+1] = mu[t] + delta[t] + rnorm(1, 0, sigma.level).}  The
  equation for the slope is
  \deqn{\delta_{t+1} = \delta_t + \eta_t \qquad \eta_t \sim
    \mathcal{N}(0, \sigma_\delta).}{%
    delta[t+1] = delta[t] + rnorm(1, 0, sigma.slope).}
  The prior distribution is on the level standard deviation
  \eqn{\sigma_\mu}{sigma.level} and the slope standard deviation
  \eqn{\sigma_\delta}{sigma.slope}.
}

\usage{
  AddLocalLinearTrend(
     state.specification = NULL,
     y,
     level.sigma.prior = NULL,
     slope.sigma.prior = NULL,
     initial.level.prior = NULL,
     initial.slope.prior = NULL,
     sdy,
     initial.y)
}

\arguments{
  \item{state.specification}{A list of state components that you wish to add to.  If
    omitted, an empty list will be assumed.  }

  \item{y}{The time series to be modeled, as a numeric vector.}

 \item{level.sigma.prior}{An object created by
   \code{\link[Boom]{SdPrior}} describing the prior distribution for
   the standard deviation of the level component.}

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

\value{ Returns a list with the elements necessary to specify a local
  linear trend state model.}

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
