


\alias{AverageExposureTime}
\name{average.exposure.time}
\title{Average exposure time}
\Rdversion{1.0}

\description{
  Computes the average exposure time for a set of Poisson process objects.
}

\usage{
AverageExposureTime(point.process.list,
                    timescale = c("days", "secs", "mins", "hours", "weeks"),
                    reduce = TRUE)
}

\arguments{

  \item{point.process.list}{Either a single \code{\link{PointProcess}}
    object, or a list of such objects.}

  \item{timescale}{timescale: The time scale to use when computing time
    differences.}

  \item{reduce}{Logical.  See below.}
}

\value{ If \code{reduce} is \code{TRUE} then a grand mean will be
  computed from the list of inputs.  Otherwise a vector containing the
  exposure time of each element in \code{point.process.list} is returned.}

\author{
  Steven L. Scott \email{steve.the.bayesian@gmail.com}
}

\examples{

start <- as.POSIXct("2012-04-12")
dates <- start + sort(runif(100)) * 3600 * 24 * 7
## 100 time points uniformly distributed between start and one week
## later.
process <- PointProcess(dates, start)

AverageExposureTime(process, timescale = "week")
AverageExposureTime(process, timescale = "secs")

second.process <- PointProcess(start + sort(runif(100)) * 3600 * 24 * 7, start)
AverageExposureTime(list(process, second.process))
AverageExposureTime(list(process, second.process), reduce = FALSE)
AverageExposureTime(list(process, second.process),
                    reduce = FALSE,
                    timescale = "hours")
}

\keyword{ts}
