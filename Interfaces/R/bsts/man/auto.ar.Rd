% Copyright 2012 Google Inc. All Rights Reserved.
% Author: stevescott@google.com (Steve Scott)

\name{auto.ar}
\alias{AddAutoAr}

\Rdversion{1.1}
\title{
  Sparse AR(p)
}

\description{ Add a sparse AR(p) process to the state distribution.  A
  sparse AR(p) is an AR(p) process with a spike and slab prior on the
  autoregression coefficients.  }

\usage{
AddAutoAr(state.specification,
          y,
          lags = 1,
          prior = NULL,
          sdy = NULL,
          ...)
}

\arguments{
  \item{state.specification}{A list of state components.  If omitted,
    an empty list is assumed.}

  \item{y}{ A numeric vector.  The time series to be modeled.  This can
    be omitted if \code{sdy} is supplied.}

  \item{lags}{The maximum number of lags ("p") to be considered in the AR(p) process.}

  \item{prior}{An object inheriting from \code{\link{SpikeSlabArPrior}}, or
    \code{NULL}.  If the latter, then a default
    \code{\link{SpikeSlabArPrior}} will be created.}

  \item{sdy}{The sample standard deviation of the time series to be
    modeled.  Used to scale the prior distribution.  This can be omitted
    if \code{y} is supplied.  }

  \item{\dots}{Extra arguments passed to \code{\link{SpikeSlabArPrior}}.}

}

\value{ Returns \code{state.specification} with an AR(p) state component
added to the end.}

\details{
  The model contributes alpha[t] to the expected value of y[t], where
  the transition equation is

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

  This model differs from the one in \code{\link{AddAr}} only in that
  some of its coefficients may be set to zero.
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
ss <- AddAutoAr(list(), y, lags = 6)

# Fit the model with knowledge with residual.sd essentially fixed at the
# true value.
model <- bsts(y, state.specification=ss, niter = 500, prior = SdPrior(residual.sd, 100000))

# Now compare the empirical ACF to the true ACF.
acf(y, lag.max = 30)
points(0:30, ARMAacf(ar = true.phi, lag.max = 30), pch = "+")
points(0:30, ARMAacf(ar = colMeans(model$AR6.coefficients), lag.max = 30))
legend("topright", leg = c("empirical", "truth", "MCMC"), pch = c(NA, "+", "o"))
}

\seealso{
  \code{\link{bsts}}.
  \code{\link[Boom]{SdPrior}}
}

\keyword{models}
