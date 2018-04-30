


\alias{density.PointProcess}
\alias{PointProcessDensity}
\name{density.point.process}
\title{Estimate the intensity of a point process}
\Rdversion{1.0}

\description{
  Nonparametric estimate of point process data intensity.
}

\usage{
  \method{density}{PointProcess}(x, from = x$start, to = x$end, ...)
}

\arguments{

  \item{x}{An object of class \code{\link{PointProcess}}}.

  \item{from}{An object convertible to \code{\link{POSIXt}} indicating
    the beginning of the range over which the intensity estimate is desired.}

  \item{to}{An object convertible to \code{\link{POSIXt}} indicating
    the end of the range over which the intensity estimate is desired.}

  \item{\dots}{Extra arguments passed to \code{\link{density}}.}

}

\value{
  An object of class \code{PointProcessDensity}.  This is an ordinary
  \code{\link{density}} object modifed to account for the fact that the
  numbers are dates.  The main use of this object is to pass it to its
  plot method.  This is often done implictly as in the examples below.
}

\author{
  Steven L. Scott \email{steve.the.bayesian@gmail.com}
}

\seealso{
  \code{\link{BoomEvents-package}}.
}

\examples{
start <- as.POSIXct("2012-04-12")
dates <- start + sort(runif(100)) * 3600 * 24 * 7
## 100 time points uniformly distributed between start and one week
## later.
process <- PointProcess(dates, start)
plot(density(process))
}
