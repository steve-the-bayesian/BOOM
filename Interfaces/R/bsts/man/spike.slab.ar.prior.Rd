\name{spike.slab.ar.prior}

\alias{SpikeSlabArPrior}

\Rdversion{1.0}

\title{
  Spike and Slab Priors for AR Processes
}

\description{ Returns a spike and slab prior for the parameters of an
  AR(p) process.
}

\usage{
SpikeSlabArPrior(
    lags,
    prior.inclusion.probabilities =
        GeometricSequence( lags, initial.value = .8, discount.factor = .8),
    prior.mean = rep(0, lags),
    prior.sd =
        GeometricSequence(lags, initial.value = .5, discount.factor = .8),
    sdy,
    prior.df = 1,
    expected.r2 = .5,
    sigma.upper.limit = Inf,
    truncate = TRUE)
}

\arguments{
  \item{lags}{A positive integer giving the maximum number of lags to
    consider.}

  \item{prior.inclusion.probabilities}{ A vector of length \code{lags}
    giving the prior probability that the corresponding AR coefficient
    is nonzero.  }

  \item{prior.mean}{A vector of length \code{lags} giving the prior mean
    of the AR coefficients.  This should almost surely stay set at zero.
  }

  \item{prior.sd}{A vector of length \code{lags} giving the prior
    standard deviations of the AR coefficients, which are modeled as
    a-priori independent of one another.}

  \item{sdy}{The sample standard deviation of the series being modeled.}

  \item{expected.r2}{The expected fraction of variation in the response
    explained by this AR proces.}

  \item{prior.df}{A positive number indicating the number of
    observations (time points) worth of weight to assign to the guess at
    \code{expected.r2}.}

  \item{sigma.upper.limit}{A positive number less than infinity
    truncates the support of the prior distribution to regions where the
    residual standard deviation is less than the specified limit.  Any
    other value indicates support over the entire positive real line.}

  \item{truncate}{If \code{TRUE} then the support of the distribution is
    truncated to the region where the AR coefficients imply a stationary
    process.  If \code{FALSE} the coefficients are unconstrained.}

}

\value{
  A list of class \code{SpikeSlabArPrior} containing the information
  needed for the underlying C++ code to instantiate this prior.
}

\author{
  Steven L. Scott \email{stevescott@google.com}
}
