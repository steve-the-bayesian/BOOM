% Copyright 2016 Google Inc. All Rights Reserved.
% Author: stevescott@google.com (Steve Scott)

\name{lognormal.prior}
\title{Lognormal Prior Distribution}
\alias{LognormalPrior}

\description{Specifies a lognormal prior distribution.}

\usage{
  LognormalPrior(mu = 0.0, sigma = 1.0, initial.value = NULL)
}

\arguments{
  \item{mu}{mean of the corresponding normal distribution.}

  \item{sigma}{standard deviation of the corresponding normal
    distribution.  WARNING: If something looks strange in your
    program, look out for SD != Variance errors.}

  \item{initial.value}{Initial value of the variable to be modeled
    (e.g. in an MCMC algorithm).  If \code{NULL} then the prior mean will
    be used.}
}

\details{
  A lognormal distribution, where log(y) ~ N(mu, sigma).  The mean
  of this distribution is exp(mu + 0.5 * sigma^2), so don't only
  focus on the mean parameter here.

}

\references{
  Gelman, Carlin, Stern, Rubin (2003), "Bayesian Data Analysis", Chapman
  and Hall.

  \url{https://en.wikipedia.org/wiki/Log-normal_distribution}
}

\author{
  Steven L. Scott \email{stevescott@google.com}
}
