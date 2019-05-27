% Copyright 2019 Steven L. Scott. All Rights Reserved.
% Author: steve.the.bayesian@gmail.com (Steve Scott)

\name{add.shared.local.level}
\alias{AddSharedLocalLevel}

\Rdversion{1.1}
\title{
  Local level trend state component
}

\description{
  Add a shared local level model to a state specification.
  The shared local level model assumes the trend is a
  multivariate random walk:

  \deqn{\alpha_{t+1} = \alpha_t + \eta_t \qquad
    \eta_{tj} \sim \mathcal{N}(0,\sigma_j).}{%
    alpha[t+1, ] = alpha[t, ] + rnorm(nfactors, 0, sigma). }

  The contribution to the mean of the observed series obeys \deqn{y_{t}
  = B \alpha_t + \epsilon_t.}{y[t, ] = B \%*\% alpha[t, ]} plus
  observation error.  Identifiability constraints ipmly that the
  observation coefficients \code{B} form a rectangular lower triangular
  matrix with diagonal 1.0.
}

\usage{
  AddSharedLocalLevel(
     state.specification,
     response,
     nfactors, 
     coefficient.prior = NULL,
     initial.state.prior = NULL,
     timestamps = NULL,
     series.id = NULL,
     sdy,
     ...)
}

\arguments{

  \item{state.specification}{A pre-existing list of state components
    that you wish to add to.  If omitted, an empty list will be assumed.
  }

  \item{response}{ The time series to be modeled. This can either be a
    matrix with rows as time and columns as series, or it can be a numeric
    vector.  If a vector is passed then \code{timestamps} and
    \code{series.id} are required.  Otherwise they are unused.}

  \item{nfactors}{The number of latent factors to include in the model.
    This is the dimension of the state for this model component.}

  \item{coefficient.prior}{Prior distribution on the observation
    coefficients.  This currently must be a
    \code{\link[Boom]{ScaledMatrixNormalPrior}}.  Expect other choices
    in the future. }
  
  \item{initial.state.prior}{An object of class
    \code{\link[Boom]{MvnPrior}}, describing the prior distribution of
    the initial state vector (at time 1).}

  \item{timestamps}{If \code{response} is in long format (i.e. a vector
    instead of a matrix) this argument is a vector of the same
    length indicating the time index to which each element of
    \code{response} belongs. }

  \item{series.id}{If \code{response} is in long format (i.e. a vector
    instead of a matrix) this argument is a vector of the same length
    indicating the time series to which each element of \code{response}
    belongs. }
  
  \item{sdy}{A vector giving the standard deviation of each series to be
    modeled.  This argument is only necessary if \code{response} cannot
    be supplied directly.}

  \item{\dots}{Extra arguments passed to
    \code{\link[BoomSpikeSlab]{ConditionalZellnerPrior}}, used to create
    a default prior for the observation coefficients when
    \code{coefficient.prior} is left as \code{NULL}.}
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
