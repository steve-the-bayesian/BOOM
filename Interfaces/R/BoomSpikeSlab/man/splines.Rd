% Copyright 2019 Steven L. Scott. All Rights Reserved.
% Author: steve.the.bayesian@gmail.com (Steve Scott)

\name{spliunes}
\Rdversion{1.1}
\alias{BsplineBasis}
\alias{MsplineBasis}
\alias{IsplineBasis}
\alias{knots.SplineBasis}
\alias{knots}

\title{
  Spline Basis Expansions
}

\description{
  Spline basis expansions of a continuous variable.
}

\usage{
  BsplineBasis(x, knots = NULL, numknots = 3)
  MsplineBasis(x, knots = NULL, numknots = 3)
  IsplineBasis(x, knots = NULL, numknots = 3)

  \method{knots}{SplineBasis}(Fn, ...)

}

\arguments{

  \item{x}{
    A numeric vector to be expanded.
  }

  \item{knots}{ A numeric vector of knots defining the expansion.  The
    smallest and largest elements in \code{knots} defines the range of
    the expansion.  These knots are (notionally) replicated infinitely
    many times.  }

  \item{numknots}{If the knot vector is \code{NULL} then create a vector
    of length \code{numknots} that partitions \code{x} into
    \code{numknots} + 1 eqiprobable segments.}

  \item{Fn}{A spline basis matrix.}

  \item{\dots}{Unused, but required to match the signature of the
    \code{knots} generic function in the \code{stats} package.}
}

\details{
  B-splines are the basis most commonly used for additive
  regression models.

  M-splines are an alternative to B-splines, but are rarely used.

  I-splines are integrated M-splines.  These are monotonic functions,
  which is useful in monotonic regression problems.  If all regression
  coefficients are positive then the resulting function is
  nondecreasing.
}

\value{
  \code{XsplineBasis} returns a matrix formed by the spline basis
  expansion of \code{x}.

  \code{knots(Fn)} returns the \code{knots} attribute of \code{Fn},
  which might be useful in a second call to the basis expansion
  function.
}

\references{
  Bsplines are described in 
  deBoor (2001), "A Practical Guide to Splines".  Springer.

  Msplines and Isplines are reviewed by Ramsay (1988), Statistical
  Science pp. 425-461.
}

\author{
  Steven L. Scott
}

\examples{

# Plot the B-spline basis for x with knots determined by 3 quantiles.
x <- sort(rnorm(1000))
basis <- BsplineBasis(x, numknots=3)
par(mfrow=c(2,3))
for(i in 1:5) plot(x, basis[, i], type="l")

# Plot the I-spline basis for x with the same knots.
basis <- IsplineBasis(x, numknots=3)
par(mfrow=c(2,3))
for(i in 1:5) plot(x, basis[, i], type="l")

# Bring you own knots...
basis <- BsplineBasis(x, knots = quantile(x, c(.2, .5, .8, .9)))
par(mfrow=c(2,3))
for(i in 1:6) plot(x, basis[, i], type="l")

knots(basis)

}

\keyword{models}
\keyword{regression}
