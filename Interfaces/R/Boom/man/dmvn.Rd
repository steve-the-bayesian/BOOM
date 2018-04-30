% Copyright 2011 Google Inc. All Rights Reserved.
% Author: steve.the.bayesian@gmail.com (Steve Scott)

\name{dmvn}
\title{Multivariate Normal Density}
\alias{dmvn}

\description{Evaluate the multivariate normal density.}

\usage{
dmvn(y, mu, sigma, siginv = NULL, ldsi = NULL, logscale = FALSE)
}

\arguments{

  \item{y}{A numeric vector or matrix containing the data whose density
    is desired.}

  \item{mu}{The mean of the distribution.  A vector.}

  \item{sigma}{The variance matrix of the distribution.  A matrix.}

  \item{siginv}{The inverse of \code{sigma}, or \code{NULL}.  If
    \code{siginv} is non-\code{NULL} then \code{sigma} will not be
    used.}

  \item{ldsi}{The log determinant of \code{siginv} or \code{NULL}.}

  \item{logscale}{Logical.  If \code{TRUE} then the density is returned
    on the log scale.  Otherwise the density is returned on the density
    scale.}

}

\value{A vector containing the density of each row of \code{y}.}

\author{
  Steven L. Scott \email{steve.the.bayesian@gmail.com}
}
