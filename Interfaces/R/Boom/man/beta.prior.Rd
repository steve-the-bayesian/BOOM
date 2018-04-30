% Copyright 2011 Google Inc. All Rights Reserved.
% Author: steve.the.bayesian@gmail.com (Steve Scott)

\name{beta.prior}
\title{Beta prior for a binomial proportion}
\alias{BetaPrior}

\description{Specifies beta prior distribution for a binomial
  probability parameter.}

\usage{
  BetaPrior(a = 1, b = 1, mean = NULL, sample.size = NULL,
            initial.value = NULL)
}

\arguments{
  \item{a}{A positive real number interpretable as a prior success count.}
  \item{b}{A positive real number interpretable as a prior failure
    count.}
  \item{mean}{A positive real number representing \code{a/(a+b)}.}
  \item{sample.size}{A positive real number representing \code{a+b}.}
  \item{initial.value}{An initial value to be used for the variable
    being modeled.  If \code{NULL} then the mean of the distribution
    will be used instead.}
}

\details{
  The distribution should be specified either with \code{a} and
  \code{b}, or with \code{mean} and \code{sample.size}.
}

\references{
  Gelman, Carlin, Stern, Rubin (2003), "Bayesian Data Analysis", Chapman
  and Hall.
}

\author{
  Steven L. Scott \email{steve.the.bayesian@gmail.com}
}
