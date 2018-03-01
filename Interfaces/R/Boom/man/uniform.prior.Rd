\name{uniform.prior}
\title{Uniform prior distribution }
\alias{UniformPrior}

\description{Specifies a uniform prior distribution on a real-valued
  scalar parameter.}

\usage{
  UniformPrior(lo = 0, hi = 1, initial.value = NULL)
}

\arguments{
  \item{lo}{The lower limit of support.}
  \item{hi}{The upper limit of support.}

  \item{initial.value}{The initial value of the parameter in question to
    use in the MCMC algorithm.  If \code{NULL} then the mean \code{(lo +
      hi)/2} is used.}  }

\references{
  Gelman, Carlin, Stern, Rubin (2003), "Bayesian Data Analysis", Chapman
  and Hall.
}

\author{
  Steven L. Scott \email{stevescott@google.com}
}
