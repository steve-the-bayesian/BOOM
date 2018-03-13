% Copyright 2012 Google Inc. All Rights Reserved.
% Author: steve.the.bayesian@gmail.com (Steve Scott)

\name{add.ar}
\alias{AddAr}

\Rdversion{1.1}
\title{
  AR(p) state component
}

\description{
  Add an AR(p) state component to the state specification.
}

\usage{
AddAr(state.specification,
      y,
      lags = 1,
      sigma.prior,
      initial.state.prior = NULL,
      sdy)
}

\arguments{
  \item{state.specification}{A list of state components.  If omitted,
    an empty list is assumed.}

  \item{y}{  A numeric vector.  The time series to be modeled.}

  \item{lags}{The number of lags ("p") in the AR(p) process.}

  \item{sigma.prior}{An object created by SdPrior.  The prior for
    the standard deviation of the process increments.}

  \item{initial.state.prior}{ An object of class MvnPrior describing the
    values of the state at time 0.  This argument can be \code{NULL},
    in which case the stationary distribution of the AR(p) process
    will be used as the initial state distribution.}

  \item{sdy}{The sample standard deviation of the time series to be
    modeled.  Used to scale the prior distribution.  This can be omitted
    if \code{y} is supplied.  }
}

\value{ Returns \code{state.specification} with an AR(p) state component
added to the end.}

\details{
  The model is

  \deqn{\alpha_{t} = \phi_1\alpha_{i, t-1} + \cdots + \phi_p
    \alpha_{t-p} + \epsilon_{t-1} \qquad
    \epsilon_t \sim \mathcal{N}(0, \sigma^2)}{%
    alpha[t] = phi[1] * alpha[t-1] + ... + phi[p] * alpha[t-p] +
    epsilon[t-1],  with epsilon[t-1] ~ N(0, sigma^2)
  }

  The state consists of the last \code{p} lags of \code{alpha}.  The
  state transition matrix has \code{phi} in its first row, ones along
  its first subdiagonal, and zeros elsewhere.  The state variance matrix
  has \code{sigma^2} in its upper left corner and is zero elsewhere.
  The observation matrix has 1 in its first element and is zero
  otherwise.

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

\examples{
n <- 100
residual.sd <- .001

# Actual values of the AR coefficients
true.phi <- c(-.7, .3, .15)
ar <- arima.sim(model = list(ar = true.phi),
                n = n,
                sd = 3)

## Layer some noise on top of the AR process.
y <- ar + rnorm(n, 0, residual.sd)
ss <- AddAr(list(), lags = 3, sigma.prior = SdPrior(3.0, 1.0))

# Fit the model with knowledge with residual.sd essentially fixed at the
# true value.
model <- bsts(y, state.specification=ss, niter = 500, prior = SdPrior(residual.sd, 100000))

# Now compare the empirical ACF to the true ACF.
acf(y, lag.max = 30)
points(0:30, ARMAacf(ar = true.phi, lag.max = 30), pch = "+")
points(0:30, ARMAacf(ar = colMeans(model$AR3.coefficients), lag.max = 30))
legend("topright", leg = c("empirical", "truth", "MCMC"), pch = c(NA, "+", "o"))
}

\seealso{
  \code{\link{bsts}}.
  \code{\link[Boom]{SdPrior}}
}

\keyword{models}
