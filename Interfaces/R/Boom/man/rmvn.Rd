% Copyright 2011 Google Inc. All Rights Reserved.
% Author: steve.the.bayesian@gmail.com (Steve Scott)

\name{rmvn}
\title{Multivariate Normal Simulation}
\alias{rmvn}

\description{Simulate draws from the multivariate normal distribution.}

\usage{
rmvn(n = 1, mu, sigma = diag(rep(1., length(mu))))
}

\arguments{
  \item{n}{The desired number of draws.}
  \item{mu}{The mean of the distribution.  A vector.}
  \item{sigma}{The variance matrix of the distribution.  A matrix.}
}

\value{ If \code{n == 1} the return value is a vector.  Otherwise it is
  a matrix with \code{n} rows and \code{length(mu)} columns.}

\details{
  Note that \code{mu} and \code{sigma} are the same for all \code{n}
  draws.  This function cannot handle separate parameters for each draw
  the way \code{\link{rnorm}} and similar functions for scalar random
  variables can.
}

\examples{
y1 <- rnorm(1, 1:3)
## y1 is a vector
y2 <- rnorm(10, 1:3)
## y2 is a matrix
}

\author{
  Steven L. Scott \email{steve.the.bayesian@gmail.com}
}
