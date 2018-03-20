% Copyright 2011 Google Inc. All Rights Reserved.
% Author: stevescott@google.com (Steve Scott)

\name{ar1.coefficient.prior}
\title{Normal prior for an AR1 coefficient}
\alias{Ar1CoefficientPrior}

\description{A (possibly truncated) Gaussian prior on the autoregression
  coefficient in an AR1 model.}

\usage{
Ar1CoefficientPrior(mu = 0, sigma = 1, force.stationary = TRUE,
    force.positive = FALSE, initial.value = mu)
}

\arguments{
  \item{mu}{The mean of the prior distribution.}

  \item{sigma}{The standard deviation of the prior distribution.}

  \item{force.stationary}{Logical.  If \code{TRUE} then the prior
    support for the AR1 coefficient will be truncated to (-1, 1).}

  \item{force.positive}{Logical.  If \code{TRUE} then the prior for the
    AR1 coefficient will be truncated so that zero support is given to
    values less than zero.}

  \item{initial.value}{The initial value of the parameter being modeled
    in the MCMC algorithm.}
}

\details{ The \code{Ar1CoefficientPrior()} syntax is preferred, as it
  more closely matches R's syntax for other constructors.}

\references{
  Gelman, Carlin, Stern, Rubin (2003), "Bayesian Data Analysis", Chapman
  and Hall.
}

\author{
  Steven L. Scott \email{stevescott@google.com}
}
