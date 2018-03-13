% Copyright 2012 Google Inc. All Rights Reserved.
% Author: stevescott@google.com (Steve Scott)

\name{mvn.diagonal.prior}
\title{diagonal MVN prior}
\alias{MvnDiagonalPrior}

\description{A multivariate normal prior distribution formed by the
  product of independent normal margins.}

\usage{
MvnDiagonalPrior(mean.vector, sd.vector)
}

\arguments{
  \item{mean.vector}{A vector giving the mean of the prior distribution.}
  \item{sd.vector}{The standard deviations of the components in the
    distribution.  I.e. the square root of the diagonal of the variance
    matrix.}
}

\references{
  Gelman, Carlin, Stern, Rubin (2003), "Bayesian Data Analysis", Chapman
  and Hall.
}

\author{
  Steven L. Scott \email{stevescott@google.com}
}
