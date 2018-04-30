


\alias{PointProcess}
\name{point.process}
\title{point process data}
\Rdversion{1.0}

\description{
  Constructor for a PointProcess data object.
}

\usage{
  PointProcess(events, start = NULL, end = NULL, group.id = NULL)
}

\arguments{

  \item{events}{A vector of event times convertible to
  \code{\link{POSIXt}}.  The event times must be sorted in increasing
  order. }

  \item{start}{An optional time (convertible to \code{\link{POSIXct}})
     for the start of the observation window.  If missing, \code{start}
     is taken to be the first element of \code{events}.}

  \item{end}{An optional time (convertible to \code{\link{POSIXct}}) for
     the end of the observation window.}

  \item{group.id}{An optional factor that can be used to create a list
     of point process objects, one for each distinct level of
     \code{group.id}, each with the same \code{start} and \code{end}.}
 }

\value{
     If \code{group.id} is \code{NULL}, then this returns an object of
     class \code{PointProcess}, which is a list with elements
     \code{events}, \code{start}, and \code{end}.  If \code{group.id} is
     non-null then a list of \code{PointProcess} objects is returned.
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
plot(process)
}
\keyword{models}
