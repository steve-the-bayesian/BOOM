\name{TimeSeriesBoxplot}
\Rdversion{1.0}
\alias{TimeSeriesBoxplot}
\title{
  Time Series Boxplots
}
\description{
  Creates a series of boxplots showing the evolution of a distribution
  over time.
}
\usage{
  TimeSeriesBoxplot(x, time, ylim = NULL, add = FALSE, ...)
}
\arguments{

  \item{x}{ A matrix where each row represents a curve (e.g. a
    simulation of a time series from a posterior distribution) and
    columns represent time.  A long time series would be a wide matrix. }

  \item{time}{A vector of class \code{\link{Date}} with lenght matching
    the number of columns in \code{x}.}

  \item{ylim}{limits for the y axis.}

  \item{add}{logical, if \code{TRUE} then add boxplots to current plot.}

  \item{\dots}{Extra arguments to pass on to \code{\link{boxplot}} }
}

\value{ Called for its side effect, which is to produce a plot on the
  current graphics device.  }

\author{
  Steven L. Scott \email{steve.the.bayesian@gmail.com}
}
\examples{
  x <- t(matrix(rnorm(1000 * 100, 1:100, 1:100), nrow=100))
  ## x has 1000 rows, and 100 columns.  Column i is N(i, i^2) noise.
  time <- as.Date("2010-01-01", format = "\%Y-\%m-\%d") + (0:99 - 50)*7
  TimeSeriesBoxplot(x, time)
}

\keyword{ hplot }
