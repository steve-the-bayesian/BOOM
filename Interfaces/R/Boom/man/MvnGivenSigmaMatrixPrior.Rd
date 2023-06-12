% Copyright 2021 Steven L. Scott. All Rights Reserved.
% Author: steve.the.bayesian@gmail.com (Steve Scott)

\name{MvnGivenSigmaMatrixPrior}
\title{Conditional Multivaraite Normal Prior Given Variance}
\alias{MvnGivenSigmaMatrixPrior}

\description{A multivaraite normal prior distribution, typically used as
the prior distribution for the mean of multivaraite normal data.  The
variance of this distribution is proportional to another parameter
"Sigma" that exists elsewhere.  Usually "Sigma" is the variance of the
data.

This distribution is the "normal" part of a "normal-inverse Wishart"
distribution.
}

\usage{
MvnGivenSigmaMatrixPrior(mean, sample.size)
}

\arguments{
  \item{mean}{A vector giving the mean of the prior distribution.}

  \item{sample.size}{A positive scalar.  The variance of the
    distribution is \code{Sigma / sample.size}.}
}

\references{
  Gelman, Carlin, Stern, Rubin (2003), "Bayesian Data Analysis", Chapman
  and Hall.
}

\author{
  Steven L. Scott \email{steve.the.bayesian@gmail.com}
}
