% Copyright 2012 Google Inc. All Rights Reserved.
% Author: stevescott@google.com (Steve Scott)

\name{dirichlet.prior}
\title{Dirichlet prior for a multinomial distribution}
\alias{DirichletPrior}

\description{Specifies Dirichlet prior for a discrete probability
  distribution.}

\usage{
 DirichletPrior(prior.counts, initial.value = NULL)
}

\arguments{
  \item{prior.counts}{A vector of positive numbers representing prior counts.}
  \item{initial.value}{The initial value in the MCMC algorithm of the
    distribution being modeled.}
}

\references{
  Gelman, Carlin, Stern, Rubin (2003), "Bayesian Data Analysis", Chapman
  and Hall.
}

\author{
  Steven L. Scott \email{stevescott@google.com}
}
