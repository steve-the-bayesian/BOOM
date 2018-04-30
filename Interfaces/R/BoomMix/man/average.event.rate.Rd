


\alias{AverageEventRate}
\name{average.event.rate}
\title{Average event rate for a point process}
\Rdversion{1.0}

\description{
  Computes the average event rate from a the list of PointProcess objects.
}

\usage{
AverageEventRate(point.process,
                 timescale = c("days", "secs", "mins", "hours", "weeks"),
                 reduce = TRUE)
}

\arguments{

  \item{point.process}{Either a single \code{\link{PointProcess}}
    object, or a list of such objects.}

  \item{timescale}{The scale of the denominator to use when computing
    the average number of events per unit of time.}

  \item{reduce}{If \code{TRUE} then a grand mean will be computed from
       the list of inputs.  If \code{FALSE} then a vector is returned
       with the average event rate for each element in the point.process
       list.  If a single \code{\link{PointProcess}} was supplied then
       this argument is irrelevant.}

}

\value{ A vector giving the average number of events per unit of time,
  on the requested time scale.  }


\author{
  Steven L. Scott \email{steve.the.bayesian@gmail.com}
}

\examples{

start <- as.POSIXct("2012-04-12")
dates <- start + sort(runif(100)) * 3600 * 24 * 7
## 100 time points uniformly distributed between start and one week
## later.
process <- PointProcess(dates, start)

AverageEventRate(process, timescale = "week")
AverageEventRate(process, timescale = "secs")

second.process <- PointProcess(start + sort(runif(100)) * 3600 * 24 * 7, start)
AverageEventRate(list(process, second.process))
AverageEventRate(list(process, second.process), reduce = FALSE)
AverageEventRate(list(process, second.process),
                 reduce = FALSE,
                 timescale = "hours")
}

\keyword{ts}
