% Copyright 2012 Google Inc. All Rights Reserved.
% Author: stevescott@google.com (Steve Scott)

\name{mvn.prior}
\title{Multivariate normal prior}
\alias{MvnPrior}

\description{A multivariate normal prior distribution.}

\usage{
MvnPrior(mean, variance)
}

\arguments{
  \item{mean}{A vector giving the mean of the prior distribution.}

  \item{variance}{A symmetric positive definite matrix giving the
  variance of the prior distribution.}

}

\references{
  Gelman, Carlin, Stern, Rubin (2003), "Bayesian Data Analysis", Chapman
  and Hall.
}

\author{
  Steven L. Scott \email{stevescott@google.com}
}
