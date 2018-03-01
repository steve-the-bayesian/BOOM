% Copyright 2011 Google Inc. All Rights Reserved.
% Author: stevescott@google.com (Steve Scott)

\name{sd.prior}
\title{Prior for a standard deviation or variance}
\alias{SdPrior}

\description{Specifies an inverse Gamma prior for a variance parameter,
  but inputs are defined in terms of a standard deviation.}

\usage{
  SdPrior(sigma.guess, sample.size = .01, initial.value = sigma.guess,
          fixed = FALSE, upper.limit = Inf)
}

\arguments{
  \item{sigma.guess}{A prior guess at the value of the standard deviation.}

  \item{sample.size}{The weight given to \code{sigma.guess}.
    Interpretable as a prior observation count.}

  \item{initial.value}{The initial value of the paramter in the MCMC
    algorithm.}

  \item{fixed}{Logical.  Some algorithms allow you to fix sigma at a
    particular value.  If \code{TRUE} then sigma will remain fixed at
    \code{initial.value}, if supported.}

  \item{upper.limit}{If positive, this is the upper limit on possible
    values of the standard deviation parameter.  Otherwise the upper
    limit is assumed infinite.  Not supported by all MCMC algorithms.}
}

\references{
  Gelman, Carlin, Stern, Rubin (2003), "Bayesian Data Analysis", Chapman
  and Hall.
}

\author{
  Steven L. Scott \email{stevescott@google.com}
}
