% Copyright 2011 Google Inc. All Rights Reserved.
% Author: stevescott@google.com (Steve Scott)

\name{markov.prior}
\title{Prior for a Markov chain}
\alias{MarkovPrior}

\description{The conjugate prior distribution for the parameters of a
  homogeneous Markov chain.  The rows in the transition probability
  matrix modeled with independent Dirichlet priors.  The distribution of
  the initial state is modeled with its own independent Dirichlet
  prior.}

\usage{
MarkovPrior(prior.transition.counts = NULL,
            prior.initial.state.counts = NULL,
            state.space.size = NULL,
            uniform.prior.value = 1)
}

\arguments{

  \item{prior.transition.counts}{A matrix of the same dimension as the
    transition probability matrix being modeled.  Entry (i, j) represents
    the "prior count" of transitions from state \code{i} to state \code{j}.}

  \item{prior.initial.state.counts}{A vector of positive numbers
    representing prior counts of initial states.}

  \item{state.space.size}{If both prior.transition.counts and
    prior.initial.state.counts are missing, then they will be filled
    with an object of dimension state.space.size where all entries are
    set to \code{uniform.prior.value}.}

  \item{uniform.prior.value}{The default value to use for entries of
    \code{prior.transition.counts} and
    \code{prior.initial.state.counts}, when they are not supplied by the
    user.}  }

\references{
  Gelman, Carlin, Stern, Rubin (2003), "Bayesian Data Analysis", Chapman
  and Hall.
}

\author{
  Steven L. Scott \email{stevescott@google.com}
}
