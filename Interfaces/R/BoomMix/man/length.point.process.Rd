


\alias{length.PointProcess}
\name{length.point.process}
\title{length of a point process}
\Rdversion{1.0}

\description{
  length of a PointProcess
}

\usage{
  \method{length}{PointProcess}(x)
}

\arguments{

  \item{x}{An object of class \code{\link{PointProcess}}.}

}

\value{
  The length of a PointProcess is the number of events that have been observed.
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
length(process) ## should be 100
}
