\name{discrete-uniform-prior}
\title{Discrete prior distributions}
\alias{DiscreteUniformPrior}
\alias{PointMassPrior}
\alias{PoissonPrior}

\description{Prior distributions over a discrete quantities.}

\usage{
PointMassPrior(location)
PoissonPrior(mean, lower.limit = 0, upper.limit = Inf)
DiscreteUniformPrior(lower.limit, upper.limit)
}

\arguments{

  \item{location}{The location of the point mass.}

  \item{mean}{The mean of the Poisson distribution.}

  \item{lower.limit}{The smallest value within the support of the
    distribution.  The prior probability for numbers less than
    \code{lower.limit} is zero. }

  \item{upper.limit}{The largest value within the support of the
    distribution.  The prior probability for numbers greater than
    \code{upper.limit} is zero. }

}

\value{ Each function returns a prior object whose class is the same as
  the function name.  All of these inherit from "DiscreteUniformPrior"
  and from "Prior".

  The \code{PoissonPrior} assumes a potentially truncated Poisson
  distribution with the given mean.
}

\author{
  Steven L. Scott \email{stevescott@google.com}
}

\examples{

## Specify an exact number of trees in a Bart model (see the BoomBart
## package).

ntrees <- PointMassPrior(200)

## Uniform prior between 50 and 100 trees, including the endpoints.
ntrees <- DiscreteUniformPrior(50, 100)

## Truncated Poisson prior, with a mean of 20, a lower endpoint of 1,
## and an upper endpoint of 50.
ntrees <- PoissonPrior(20, 1, 50)

}
